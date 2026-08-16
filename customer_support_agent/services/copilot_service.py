from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - exercised indirectly in environments without the extra dependency
    ChatGroq = None

from customer_support_agent.core.settings import Settings
from customer_support_agent.integrations.memory.langmem_store import CustomerMemoryStore
from customer_support_agent.integrations.rag.chroma_kb import KnowledgeBaseService
from customer_support_agent.integrations.tools import get_support_tools


class SupportCopilot:
    def __init__(self, settings: Settings):
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ API Key is missing. Add it in .env before generating drafts."
            )

        self._settings = settings
        self._llm = self._build_llm()
        self._tools = get_support_tools()
        self._memory_error: str | None = None

        try:
            self.memory = CustomerMemoryStore(settings=settings, llm=self._llm)
        except Exception as exc:
            self._memory_error = str(exc)
        self.rag = KnowledgeBaseService(settings=settings)

    def _build_llm(self) -> Any:
        if ChatGroq is None:
            return _DeterministicLLM(
                model=self._settings.groq_model,
                temperature=self._settings.llm_temperature,
            )
        return ChatGroq(
            model=self._settings.groq_model,
            groq_api_key=self._settings.groq_api_key,
            temperature=self._settings.llm_temperature,
        )

    def generate_draft(self, ticket: dict[str, Any], customer: dict[str, Any]) -> dict[str, Any]:
        query = f"{ticket['subject']}\n{ticket['description']}"
        customer_email = customer["email"]

        memory_hits = self._search_memory_scopes(
            query=query,
            customer_email=customer_email,
            customer_company=customer.get("company"),
            limit=self._settings.mem0_top_k,
        )
        kb_hits = self.rag.search(query=query, top_k=self._settings.rag_top_k)
        tool_calls = self._run_support_tools(ticket=ticket, customer=customer)

        draft_text = self._generate_text(
            system_prompt=self._build_system_prompt(memory_hits=memory_hits, kb_hits=kb_hits),
            user_prompt=self._build_user_prompt(ticket=ticket, customer=customer),
            tool_calls=tool_calls,
        )

        used_fallback = False
        if not draft_text:
            draft_text = self._fallback_generate_text(
                ticket=ticket,
                customer=customer,
                memory_hits=memory_hits,
                kb_hits=kb_hits,
                tool_calls=tool_calls,
            )
            used_fallback = True
        if not draft_text:
            draft_text = self._deterministic_fallback(
                ticket=ticket,
                customer=customer,
                tool_calls=tool_calls,
            )
            used_fallback = True

        context_used = self._build_context(
            ticket=ticket,
            customer=customer,
            memory_hits=memory_hits,
            kb_hits=kb_hits,
            tool_calls=tool_calls,
        )
        if self._memory_error:
            context_used.setdefault("errors", []).append(f"Memory disabled: {self._memory_error}")
        if used_fallback:
            context_used.setdefault("errors", []).append(
                "Primary model response was empty; fallback synthesis was used."
            )
        context_used["agent_runtime"] = "direct_llm_with_local_tools"

        return {
            "draft": draft_text,
            "context_used": context_used,
        }

    def save_accepted_resolution(
        self,
        customer_email: str,
        customer_company: str | None,
        ticket_subject: str,
        ticket_description: str,
        draft_content: str,
        context_used: dict[str, Any] | None = None,
    ) -> None:
        entity_links = self._extract_entity_links(
            ticket_subject=ticket_subject,
            ticket_description=ticket_description,
            draft_content=draft_content,
            context_used=context_used or {},
        )
        for scope_user_id in self._memory_scope_ids(
            customer_email=customer_email,
            customer_company=customer_company,
        ):
            self.memory.add_resolution(
                user_id=scope_user_id,
                ticket_subject=ticket_subject,
                ticket_description=ticket_description,
                accepted_draft=draft_content,
                entity_links=entity_links,
            )

    def list_customer_memories(
        self,
        customer_email: str,
        customer_company: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        scope_user_ids = self._memory_scope_ids(
            customer_email=customer_email,
            customer_company=customer_company,
        )
        raw_hits: list[dict[str, Any]] = []
        for scope_user_id in scope_user_ids:
            hits = self.memory.list_memories(user_id=scope_user_id, limit=max(1, limit))
            raw_hits.extend(self._annotate_memory_scope(hits=hits, scope_user_id=scope_user_id))
        return self._dedupe_memory_hits(raw_hits, limit=max(1, limit))

    def search_customer_memories(
        self,
        customer_email: str,
        query: str,
        customer_company: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self._search_memory_scopes(
            query=query,
            customer_email=customer_email,
            customer_company=customer_company,
            limit=limit,
        )

    def _search_memory_scopes(
        self,
        query: str,
        customer_email: str,
        customer_company: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        per_scope_limit = max(1, limit)
        scope_user_ids = self._memory_scope_ids(
            customer_email=customer_email,
            customer_company=customer_company,
        )
        raw_hits: list[dict[str, Any]] = []
        for scope_user_id in scope_user_ids:
            hits = self.memory.search(query=query, user_id=scope_user_id, limit=per_scope_limit)
            raw_hits.extend(self._annotate_memory_scope(hits=hits, scope_user_id=scope_user_id))
        return self._dedupe_memory_hits(raw_hits, limit=per_scope_limit * len(scope_user_ids))

    def _run_support_tools(
        self,
        ticket: dict[str, Any],
        customer: dict[str, Any],
    ) -> list[dict[str, Any]]:
        customer_email = str(customer.get("email") or "").strip()
        if not customer_email:
            return []

        lower_text = f"{ticket.get('subject', '')}\n{ticket.get('description', '')}".lower()
        selected_tools: list[tuple[str, dict[str, Any]]] = []
        if any(keyword in lower_text for keyword in ("policy", "plan", "coverage", "sla")):
            selected_tools.append(("lookup_customer_plan", {"customer_email": customer_email}))
        if any(keyword in lower_text for keyword in ("claim", "issue", "ticket", "repeat", "again")):
            selected_tools.append(("lookup_open_ticket_load", {"customer_email": customer_email}))

        tool_calls: list[dict[str, Any]] = []
        for tool_name, arguments in selected_tools:
            tool_func = next((tool for tool in self._tools if tool.name == tool_name), None)
            if tool_func is None:
                continue
            try:
                raw_output = tool_func.invoke(arguments)
                parsed_output, output_text = self._parse_tool_output(raw_output)
                tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "tool_call_id": None,
                        "arguments": arguments,
                        "status": "ok",
                        "summary": self._tool_summary(parsed_output, output_text),
                        "output": parsed_output,
                        "output_text": output_text,
                    }
                )
            except Exception as exc:
                tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "tool_call_id": None,
                        "arguments": arguments,
                        "status": "error",
                        "summary": str(exc),
                        "output": None,
                        "output_text": str(exc),
                    }
                )
        return tool_calls

    def _generate_text(self, system_prompt: str, user_prompt: str, tool_calls: list[dict[str, Any]]) -> str:
        tool_lines = [
            f"- {item.get('tool_name')}: {item.get('summary') or item.get('output_text', '')}"
            for item in tool_calls
        ]
        tool_context = "\n".join(tool_lines) if tool_lines else "- No tool lookups were needed."
        response = self._llm.invoke(
            [
                SystemMessage(content=f"{system_prompt}\n\nTool Findings:\n{tool_context}"),
                HumanMessage(content=user_prompt),
            ]
        )
        return self._extract_content(response).strip()

    def _memory_scope_ids(self, customer_email: str, customer_company: str | None) -> list[str]:
        scope_user_ids = [customer_email.strip().lower()]
        company_scope = self._company_scope_user_id(customer_company)
        if company_scope:
            scope_user_ids.append(company_scope)
        return self._unique_ordered(scope_user_ids)

    @staticmethod
    def _company_scope_user_id(customer_company: str | None) -> str | None:
        if not customer_company:
            return None
        lowered = customer_company.strip().lower()
        if not lowered:
            return None
        normalized = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
        if not normalized:
            return None
        return f"company::{normalized}"

    @staticmethod
    def _annotate_memory_scope(
        hits: list[dict[str, Any]],
        scope_user_id: str,
    ) -> list[dict[str, Any]]:
        annotated: list[dict[str, Any]] = []
        scope = "company" if scope_user_id.startswith("company::") else "customer"
        for hit in hits:
            item = dict(hit)
            metadata = dict(item.get("metadata") or {})
            metadata.setdefault("scope", scope)
            metadata.setdefault("scope_user_id", scope_user_id)
            item["metadata"] = metadata
            annotated.append(item)
        return annotated

    @staticmethod
    def _dedupe_memory_hits(hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            memory_text = str(hit.get("memory", "")).strip()
            if not memory_text:
                continue
            key = memory_text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
            if len(deduped) >= max(1, limit):
                break
        return deduped

    @staticmethod
    def _extract_content(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)

    @staticmethod
    def _format_memory(memory_hits: list[dict[str, Any]]) -> str:
        if not memory_hits:
            return "- No prior customer memories found."
        return "\n".join(f"- {item.get('memory', '').strip()}" for item in memory_hits)

    @staticmethod
    def _format_kb(kb_hits: list[dict[str, Any]]) -> str:
        if not kb_hits:
            return "- No relevant knowledge-base chunks found."
        return "\n".join(
            f"- [{item.get('source', 'unknown')}] {item.get('content', '').strip()}"
            for item in kb_hits
        )

    def _build_system_prompt(self, memory_hits: list[dict[str, Any]], kb_hits: list[dict[str, Any]]) -> str:
        return (
            "You are an AI copilot for insurance claims adjusters handling auto claims. "
            "Write concise, factual, and actionable coverage recommendations. "
            "Use any supplied tool findings as facts.\n\n"
            "Customer Memory Context:\n"
            f"{self._format_memory(memory_hits)}\n\n"
            "Knowledge Base Context:\n"
            f"{self._format_kb(kb_hits)}\n\n"
            "Output rules:\n"
            "1) Provide a recommended coverage decision with reasoning and confidence.\n"
            "2) Include clear next steps and required documents if information is incomplete.\n"
            "3) Reference KB and tool facts when relevant, without chain-of-thought.\n"
            "4) Never present a denial as an autonomous final decision; a licensed adjuster must approve.\n"
            "5) Keep response under 220 words unless additional detail is necessary."
        )

    @staticmethod
    def _build_user_prompt(ticket: dict[str, Any], customer: dict[str, Any]) -> str:
        return (
            f"Customer: {customer.get('name') or 'Unknown'} ({customer['email']})\n"
            f"Company: {customer.get('company') or 'Unknown'}\n"
            f"Ticket Subject: {ticket['subject']}\n"
            f"Ticket Priority: {ticket.get('priority', 'medium')}\n"
            f"Ticket Description:\n{ticket['description']}\n\n"
            "Create a draft response for the support agent."
        )

    @staticmethod
    def _parse_tool_output(raw_output: Any) -> tuple[dict[str, Any] | None, str]:
        if isinstance(raw_output, dict):
            return raw_output, json.dumps(raw_output)
        output_text = str(raw_output)
        try:
            parsed = json.loads(output_text)
            if isinstance(parsed, dict):
                return parsed, output_text
        except json.JSONDecodeError:
            pass
        return None, output_text

    @staticmethod
    def _tool_summary(parsed_output: dict[str, Any] | None, output_text: str) -> str:
        if parsed_output and parsed_output.get("summary"):
            return str(parsed_output["summary"])
        return output_text

    def _build_context(
        self,
        ticket: dict[str, Any],
        customer: dict[str, Any],
        memory_hits: list[dict[str, Any]],
        kb_hits: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        knowledge_sources = self._unique_ordered(
            [str(item.get("source")) for item in kb_hits if item.get("source")]
        )
        tool_errors = [item for item in tool_calls if item.get("status") != "ok"]
        return {
            "version": 2,
            "ticket": {
                "id": ticket.get("id"),
                "subject": ticket.get("subject"),
                "priority": ticket.get("priority"),
                "status": ticket.get("status"),
            },
            "customer": {
                "id": customer.get("id"),
                "email": customer.get("email"),
                "name": customer.get("name"),
                "company": customer.get("company"),
            },
            "signals": {
                "memory_hit_count": len(memory_hits),
                "knowledge_hit_count": len(kb_hits),
                "tool_call_count": len(tool_calls),
                "tool_error_count": len(tool_errors),
                "knowledge_sources": knowledge_sources,
            },
            "highlights": {
                "memory": [self._trim_text(item.get("memory", "")) for item in memory_hits[:3]],
                "knowledge": [
                    self._trim_text(f"[{item.get('source', 'unknown')}] {item.get('content', '')}")
                    for item in kb_hits[:3]
                ],
                "tools": [self._trim_text(item.get("summary", "")) for item in tool_calls[:3]],
            },
            "memory_hits": memory_hits,
            "knowledge_hits": kb_hits,
            "tool_calls": tool_calls,
        }

    @staticmethod
    def _unique_ordered(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    @staticmethod
    def _trim_text(text: Any, limit: int = 180) -> str:
        clean = str(text or "").strip()
        if len(clean) <= limit:
            return clean
        return f"{clean[: limit - 3]}..."

    def _extract_entity_links(
        self,
        ticket_subject: str,
        ticket_description: str,
        draft_content: str,
        context_used: dict[str, Any],
    ) -> list[str]:
        merged_text = f"{ticket_subject}\n{ticket_description}\n{draft_content}"
        merged_lower = merged_text.lower()
        links: list[str] = []

        endpoints = re.findall(r"/[a-zA-Z0-9][a-zA-Z0-9/_-]{2,}", merged_text)
        for endpoint in self._unique_ordered(endpoints)[:3]:
            links.append(f"endpoint:{endpoint}")

        status_codes = re.findall(r"\b([45]\d\d)\b", merged_text)
        for code in self._unique_ordered(status_codes)[:4]:
            links.append(f"http_status:{code}")

        regions = [
            ("EU", [" eu ", "europe", "emea"]),
            ("US", [" us ", "united states", "na "]),
            ("APAC", [" apac ", "asia pacific"]),
            ("India", [" india ", " in "]),
        ]
        padded = f" {merged_lower} "
        for region, markers in regions:
            if any(marker in padded for marker in markers):
                links.append(f"region:{region}")

        integrations = ["shopify", "stripe", "salesforce", "slack", "quickbooks", "hubspot", "zendesk"]
        for integration in integrations:
            if integration in merged_lower:
                links.append(f"integration:{integration}")

        for tool_call in context_used.get("tool_calls", []):
            output = tool_call.get("output") or {}
            details = output.get("details") if isinstance(output, dict) else None
            if not isinstance(details, dict):
                continue
            plan = details.get("plan_tier")
            if plan:
                links.append(f"plan:{plan}")
            risk = details.get("risk_level")
            if risk:
                links.append(f"billing_risk:{risk}")

        return self._unique_ordered([item for item in links if item])[:12]

    def _fallback_generate_text(
        self,
        ticket: dict[str, Any],
        customer: dict[str, Any],
        memory_hits: list[dict[str, Any]],
        kb_hits: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
    ) -> str:
        tool_summaries = [
            self._trim_text(item.get("summary") or item.get("output_text", ""))
            for item in tool_calls
            if item.get("summary") or item.get("output_text")
        ]
        memory_summaries = [self._trim_text(item.get("memory", "")) for item in memory_hits[:3]]
        kb_summaries = [
            self._trim_text(f"[{item.get('source', 'unknown')}] {item.get('content', '')}")
            for item in kb_hits[:3]
        ]

        fallback_system = (
            "You are an AI support copilot. Produce only the final customer-facing draft reply."
        )
        fallback_user = (
            f"Customer: {customer.get('name') or 'Unknown'} ({customer.get('email', 'unknown')})\n"
            f"Company: {customer.get('company') or 'Unknown'}\n"
            f"Ticket subject: {ticket.get('subject', '')}\n"
            f"Ticket description: {ticket.get('description', '')}\n\n"
            "Memory highlights:\n"
            f"{chr(10).join('- ' + item for item in memory_summaries) if memory_summaries else '- none'}\n\n"
            "Knowledge highlights:\n"
            f"{chr(10).join('- ' + item for item in kb_summaries) if kb_summaries else '- none'}\n\n"
            "Tool findings:\n"
            f"{chr(10).join('- ' + item for item in tool_summaries) if tool_summaries else '- none'}\n\n"
            "Write a concise, empathetic draft with clear next steps."
        )

        try:
            response = self._llm.invoke(
                [
                    SystemMessage(content=fallback_system),
                    HumanMessage(content=fallback_user),
                ]
            )
            return self._extract_content(response).strip()
        except Exception:
            return ""

    def _deterministic_fallback(
        self,
        ticket: dict[str, Any],
        customer: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> str:
        customer_name = customer.get("name") or customer.get("email") or "there"
        best_tool_summary = ""
        for item in tool_calls:
            summary = str(item.get("summary") or "").strip()
            if summary:
                best_tool_summary = summary
                break

        action_line = (
            best_tool_summary
            if best_tool_summary
            else "Our support team is reviewing your account and issue details now."
        )

        return (
            f"Hi {customer_name},\n\n"
            f"Thanks for reaching out about \"{ticket.get('subject', 'your issue')}\". "
            "I understand how disruptive this can be.\n\n"
            f"{action_line}\n\n"
            "Next, we will continue investigating and share an update with concrete steps shortly.\n\n"
            "Best,\nSupport Team"
        )


class _DeterministicLLM:
    def __init__(self, model: str, temperature: float) -> None:
        self.model = model
        self.temperature = temperature

    def invoke(self, messages: list[Any]) -> Any:
        content_lines = [getattr(message, "content", str(message)) for message in messages]
        prompt = "\n".join(content_lines)
        summary = prompt.splitlines()[-1].strip() if prompt else "Support review"
        return SimpleNamespace(
            content=(
                "Fallback support draft generated locally.\n"
                f"Model: {self.model}\n"
                f"Prompt summary: {summary}"
            )
        )
