"""
EventHive AI — FastAPI backend with genuine multi-agent orchestration.

Workflow:
  User request → Event Manager (briefs) → Planning / Budget / Logistics in parallel
  → Python validation + budget reconciliation → Event Manager (final plan)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import traceback
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()
# On Vercel, environment variables are injected directly (set in Project
# Settings → Environment Variables) — the .env file is only for local runs.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("eventhive")

GROK_API_KEY = (os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or "").strip()
GROK_API_BASE = (os.getenv("GROK_API_BASE") or "https://api.x.ai/v1").rstrip("/")
GROK_MODEL = (os.getenv("GROK_MODEL") or "grok-4-fast").strip()
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS") or "90")
ORCHESTRATION_TIMEOUT_SECONDS = float(os.getenv("ORCHESTRATION_TIMEOUT_SECONDS") or "240")

PLACEHOLDER_KEYS = {"", "your_actual_api_key", "changeme", "replace_me"}


def api_key_configured() -> bool:
    key = GROK_API_KEY.strip()
    return bool(key) and key.lower() not in PLACEHOLDER_KEYS


def parse_cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500",
    )
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["http://localhost:5500"]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class EventRequest(BaseModel):
    event_type: str = Field(..., min_length=2, max_length=200)
    attendees: int = Field(..., gt=0, le=1_000_000)
    date: str = Field(..., min_length=8, max_length=32)
    location: str = Field(..., min_length=2, max_length=300)
    budget: float = Field(..., ge=0)
    currency: str = Field(default="PKR", min_length=1, max_length=12)
    goals: str = Field(default="", max_length=4000)
    special_requirements: str = Field(default="", max_length=4000)

    @field_validator("event_type", "location", "currency", "goals", "special_requirements")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("currency")
    @classmethod
    def currency_upper(cls, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z]", "", value).upper()
        if len(cleaned) < 3:
            raise ValueError("Currency must be a valid code such as PKR.")
        return cleaned[:8]

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                parsed = datetime.strptime(text, fmt).date()
                return parsed.isoformat()
            except ValueError:
                continue
        raise ValueError("Date must be in YYYY-MM-DD format.")


class ScheduleItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    time: str = ""
    title: str
    description: str = ""


class NamedItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    description: str = ""


class DeadlineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    item: str
    due: str = ""
    owner: str = ""


class PreparationTask(BaseModel):
    model_config = ConfigDict(extra="ignore")
    task: str
    when: str = ""
    notes: str = ""


class PrioritizedTask(BaseModel):
    model_config = ConfigDict(extra="ignore")
    priority: int = 3
    task: str
    owner_role: str = ""


class PlanningOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schedule: list[ScheduleItem] = Field(default_factory=list)
    activities: list[NamedItem] = Field(default_factory=list)
    deadlines: list[DeadlineItem] = Field(default_factory=list)
    preparation_tasks: list[PreparationTask] = Field(default_factory=list)
    prioritized_checklist: list[PrioritizedTask] = Field(default_factory=list)


class BudgetLineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    estimated_amount: float = 0
    notes: str = ""

    @field_validator("estimated_amount", mode="before")
    @classmethod
    def coerce_amount(cls, value: Any) -> float:
        if value is None or value == "":
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).replace(",", "").strip()
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return 0.0
        return float(match.group(0))


class BudgetCategory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    items: list[BudgetLineItem] = Field(default_factory=list)
    notes: str = ""


class BudgetOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    categories: list[BudgetCategory] = Field(default_factory=list)
    cost_saving_suggestions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LogisticsEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    item: str
    detail: str = ""
    assumption: bool = False
    assumption_note: str = ""


class LogisticsOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    venue_requirements: list[LogisticsEntry] = Field(default_factory=list)
    seating_and_capacity: list[LogisticsEntry] = Field(default_factory=list)
    equipment: list[LogisticsEntry] = Field(default_factory=list)
    catering: list[LogisticsEntry] = Field(default_factory=list)
    transportation: list[LogisticsEntry] = Field(default_factory=list)
    registration: list[LogisticsEntry] = Field(default_factory=list)
    setup_checklist: list[LogisticsEntry] = Field(default_factory=list)
    event_day_operations: list[LogisticsEntry] = Field(default_factory=list)


class ManagerBriefs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    understood_request: str = ""
    extracted_requirements: dict[str, Any] = Field(default_factory=dict)
    planning_brief: str
    budget_brief: str
    logistics_brief: str
    open_questions: list[str] = Field(default_factory=list)


class FinalPlanOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    event_title: str = ""
    executive_summary: str = ""
    coordination_summary: str = ""
    conflicts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)


class BudgetReconciliation(BaseModel):
    currency: str
    available_budget: float
    total_allocated: float
    remaining: float
    over_budget: bool
    overrun_amount: float
    category_totals: list[dict[str, Any]]
    notes: list[str]
    disclaimer: str = (
        "All amounts are planning estimates, not verified vendor quotations."
    )


class AgentResult(BaseModel):
    name: str
    role: str
    status: Literal["pending", "working", "completed", "failed"]
    output: dict[str, Any] | None = None
    error: str | None = None


class CreateEventResponse(BaseModel):
    success: bool
    event: dict[str, Any]
    agents: dict[str, AgentResult]
    final_plan: dict[str, Any] | None = None
    budget_reconciliation: BudgetReconciliation | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    workflow: list[dict[str, str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON helpers and budget math (deterministic)
# ---------------------------------------------------------------------------


def extract_json_object(text: str) -> dict[str, Any]:
    if not text or not str(text).strip():
        raise ValueError("The model returned an empty response.")
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("JSON root must be an object.")
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Could not parse structured JSON from the model response.")
        parsed = json.loads(cleaned[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("JSON root must be an object.")
        return parsed


def as_plain(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def reconcile_budget(budget_output: BudgetOutput, available: float, currency: str) -> BudgetReconciliation:
    notes: list[str] = []
    category_totals: list[dict[str, Any]] = []
    total_allocated = 0.0

    for category in budget_output.categories:
        cat_total = 0.0
        items: list[dict[str, Any]] = []
        for item in category.items:
            amount = round(float(item.estimated_amount), 2)
            if amount < 0:
                notes.append(f"Negative amount for '{item.name}' in '{category.name}' was treated as 0.")
                amount = 0.0
            cat_total += amount
            items.append(
                {
                    "name": item.name,
                    "estimated_amount": amount,
                    "notes": item.notes,
                }
            )
        cat_total = round(cat_total, 2)
        total_allocated += cat_total
        category_totals.append(
            {
                "name": category.name,
                "allocated": cat_total,
                "items": items,
                "notes": category.notes,
            }
        )

    total_allocated = round(total_allocated, 2)
    remaining = round(available - total_allocated, 2)
    over_budget = total_allocated > available + 0.009
    overrun_amount = round(max(0.0, total_allocated - available), 2)

    if over_budget:
        notes.append(
            f"Estimated allocations exceed the available budget by {currency} {overrun_amount:,.2f}. "
            "Do not treat this as an approved spend plan."
        )
    if not budget_output.categories:
        notes.append("Budget agent returned no categories; allocations could not be itemized.")

    notes.extend(budget_output.warnings)
    notes.append("Figures are estimates only and require vendor confirmation.")

    return BudgetReconciliation(
        currency=currency,
        available_budget=round(available, 2),
        total_allocated=total_allocated,
        remaining=remaining,
        over_budget=over_budget,
        overrun_amount=overrun_amount,
        category_totals=category_totals,
        notes=notes,
    )


def event_brief(event: EventRequest) -> str:
    return (
        f"Event type: {event.event_type}\n"
        f"Date: {event.date}\n"
        f"Location: {event.location}\n"
        f"Attendees: {event.attendees}\n"
        f"Budget: {event.currency} {event.budget:,.2f}\n"
        f"Goals: {event.goals or 'Not specified'}\n"
        f"Special requirements: {event.special_requirements or 'Not specified'}\n"
        "Constraints: Do not invent confirmed vendors, bookings, quotations, or verified venue capacity. "
        "Label estimates and assumptions clearly."
    )


# ---------------------------------------------------------------------------
# Grok HTTP client
# ---------------------------------------------------------------------------


class GrokClient:
    def __init__(self) -> None:
        self.base_url = GROK_API_BASE
        self.model = GROK_MODEL
        self.timeout = httpx.Timeout(
            connect=15.0,
            read=AGENT_TIMEOUT_SECONDS,
            write=15.0,
            pool=15.0,
        )

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not api_key_configured():
            raise RuntimeError(
                "GROK_API_KEY is missing. Add your key to the backend .env file or hosting environment variables."
            )

        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        url = f"{self.base_url}/chat/completions"
        client_timeout = httpx.Timeout(
            connect=15.0,
            read=timeout or AGENT_TIMEOUT_SECONDS,
            write=15.0,
            pool=15.0,
        )

        async with httpx.AsyncClient(timeout=client_timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException as exc:
                raise TimeoutError("The Grok API request timed out.") from exc
            except httpx.RequestError as exc:
                raise RuntimeError(
                    "Could not reach the Grok API. Check GROK_API_BASE and your network connection."
                ) from exc

        if response.status_code == 401:
            raise RuntimeError("Grok API authentication failed. Check GROK_API_KEY.")
        if response.status_code == 429:
            raise RuntimeError("Grok API rate limit reached. Wait a moment and retry.")
        if response.status_code >= 400:
            detail = _safe_error_body(response)
            raise RuntimeError(f"Grok API error ({response.status_code}): {detail}")

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("Grok API returned a non-JSON response.") from exc

        content = _message_content(body)
        try:
            return extract_json_object(content)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "The model did not return valid JSON. The agent result cannot be used."
            ) from exc


def _message_content(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("Grok API returned no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text") or "")
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)
    if not isinstance(content, str):
        raise RuntimeError("Grok API response did not include text content.")
    return content


def _safe_error_body(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)[:400]
            if isinstance(err, str):
                return err[:400]
            return str(data.get("message") or data)[:400]
    except Exception:
        pass
    return (response.text or "Unknown error")[:400]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


JSON_RULES = (
    "Return ONLY valid JSON. No markdown, no commentary, no trailing text. "
    "Never invent confirmed bookings, vendor quotes, or verified capacity. "
    "Use estimates and mark assumptions."
)


class EventManagerAgent:
    name = "event_manager"
    role = "Orchestrator"

    def __init__(self, client: GrokClient) -> None:
        self.client = client

    async def prepare_briefs(self, event: EventRequest) -> ManagerBriefs:
        system = (
            "You are the Event Manager for EventHive AI. You orchestrate specialized agents. "
            "You do not produce the full plan yourself. You understand the request, extract requirements, "
            "and write focused task briefs for Planning, Budget, and Logistics agents. "
            + JSON_RULES
        )
        user = (
            "Analyze this event request and prepare delegation briefs.\n\n"
            f"{event_brief(event)}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "understood_request": "short restatement",\n'
            '  "extracted_requirements": {"must_haves": [], "constraints": [], "success_criteria": []},\n'
            '  "planning_brief": "specific instructions for the Planning Agent",\n'
            '  "budget_brief": "specific instructions for the Budget Agent, including currency and ceiling",\n'
            '  "logistics_brief": "specific instructions for the Logistics Agent",\n'
            '  "open_questions": ["unresolved questions"]\n'
            "}\n"
            "Give each agent a distinct responsibility. Do not write their full outputs."
        )
        data = await self.client.complete_json(
            system_prompt=system,
            user_prompt=user,
            temperature=0.2,
        )
        return ManagerBriefs.model_validate(data)

    async def compose_final_plan(
        self,
        event: EventRequest,
        briefs: ManagerBriefs,
        planning: PlanningOutput,
        logistics: LogisticsOutput,
        reconciliation: BudgetReconciliation,
        specialist_warnings: list[str],
    ) -> FinalPlanOutput:
        system = (
            "You are the Event Manager for EventHive AI. Specialized agents have already produced outputs. "
            "Reconcile them into one coordinated event plan. Check conflicts, missing information, and assumptions. "
            "Do not redo their work from scratch. Do not hide budget overruns. "
            + JSON_RULES
        )
        user = (
            "Create the final coordinated plan from these actual agent results.\n\n"
            f"ORIGINAL REQUEST\n{event_brief(event)}\n\n"
            f"MANAGER UNDERSTANDING\n{briefs.understood_request}\n"
            f"Open questions from briefing: {json.dumps(briefs.open_questions)}\n\n"
            f"PLANNING AGENT OUTPUT\n{json.dumps(as_plain(planning), ensure_ascii=False)}\n\n"
            f"LOGISTICS AGENT OUTPUT\n{json.dumps(as_plain(logistics), ensure_ascii=False)}\n\n"
            "BUDGET RECONCILIATION (computed in Python — trust these numbers over any model arithmetic)\n"
            f"{json.dumps(as_plain(reconciliation), ensure_ascii=False)}\n\n"
            f"SPECIALIST WARNINGS\n{json.dumps(specialist_warnings)}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "event_title": "string",\n'
            '  "executive_summary": "1-3 paragraphs",\n'
            '  "coordination_summary": "how the three agents were combined, conflicts found, and remaining gaps",\n'
            '  "conflicts": ["string"],\n'
            '  "assumptions": ["string"],\n'
            '  "unresolved_questions": ["string"],\n'
            '  "recommended_next_steps": ["string"]\n'
            "}\n"
            "If the budget is over allocated, say so plainly and suggest reductions. "
            "Every recommendation that depends on unconfirmed details must be labeled as an assumption."
        )
        data = await self.client.complete_json(
            system_prompt=system,
            user_prompt=user,
            temperature=0.25,
        )
        return FinalPlanOutput.model_validate(data)


class PlanningAgent:
    name = "planning"
    role = "Schedule, activities, and preparation"

    def __init__(self, client: GrokClient) -> None:
        self.client = client

    async def run(self, event: EventRequest, brief: str) -> PlanningOutput:
        system = (
            "You are the Planning Agent for EventHive AI. You only plan timeline, activities, deadlines, "
            "and preparation tasks. You do not own budget totals or venue procurement. "
            + JSON_RULES
        )
        user = (
            f"EVENT\n{event_brief(event)}\n\n"
            f"TASK FROM EVENT MANAGER\n{brief}\n\n"
            "Produce a coherent event-day flow plus a preparation timeline leading up to the date. "
            "JSON schema:\n"
            "{\n"
            '  "schedule": [{"time": "HH:MM or window", "title": "string", "description": "string"}],\n'
            '  "activities": [{"name": "string", "description": "string"}],\n'
            '  "deadlines": [{"item": "string", "due": "relative or date", "owner": "role"}],\n'
            '  "preparation_tasks": [{"task": "string", "when": "string", "notes": "string"}],\n'
            '  "prioritized_checklist": [{"priority": 1, "task": "string", "owner_role": "string"}]\n'
            "}\n"
            "priority 1 is highest. Include enough items for a real hackathon-style demonstration."
        )
        data = await self.client.complete_json(
            system_prompt=system,
            user_prompt=user,
            temperature=0.4,
        )
        return PlanningOutput.model_validate(data)


class BudgetAgent:
    name = "budget"
    role = "Estimates and allocations"

    def __init__(self, client: GrokClient) -> None:
        self.client = client

    async def run(self, event: EventRequest, brief: str) -> BudgetOutput:
        system = (
            "You are the Budget Agent for EventHive AI. You propose estimated allocations only. "
            "Never present numbers as confirmed quotes. Use the user's currency. "
            "Do not hide overruns: if needs exceed the ceiling, still itemize realistic estimates. "
            "Python will compute totals. "
            + JSON_RULES
        )
        user = (
            f"EVENT\n{event_brief(event)}\n\n"
            f"TASK FROM EVENT MANAGER\n{brief}\n\n"
            f"Available ceiling: {event.currency} {event.budget:,.2f}. "
            "Itemize categories typical for this event (venue/space support, catering, equipment, "
            "registration, prizes, staffing, contingency, etc. as relevant). "
            "estimated_amount must be a number with no currency symbols.\n"
            "JSON schema:\n"
            "{\n"
            '  "categories": [{"name": "string", "notes": "string", "items": '
            '[{"name": "string", "estimated_amount": 0, "notes": "estimate only"}]}],\n'
            '  "cost_saving_suggestions": ["string"],\n'
            '  "warnings": ["string"]\n'
            "}"
        )
        data = await self.client.complete_json(
            system_prompt=system,
            user_prompt=user,
            temperature=0.2,
        )
        return BudgetOutput.model_validate(data)


class LogisticsAgent:
    name = "logistics"
    role = "Venue, operations, and setup"

    def __init__(self, client: GrokClient) -> None:
        self.client = client

    async def run(self, event: EventRequest, brief: str) -> LogisticsOutput:
        system = (
            "You are the Logistics Agent for EventHive AI. Cover venue needs, seating, equipment, "
            "catering operations, transport, registration, setup, and event-day operations. "
            "Set assumption=true whenever a recommendation is unverified. "
            + JSON_RULES
        )
        user = (
            f"EVENT\n{event_brief(event)}\n\n"
            f"TASK FROM EVENT MANAGER\n{brief}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "venue_requirements": [{"item": "", "detail": "", "assumption": true, "assumption_note": ""}],\n'
            '  "seating_and_capacity": [{"item": "", "detail": "", "assumption": true, "assumption_note": ""}],\n'
            '  "equipment": [{"item": "", "detail": "", "assumption": true, "assumption_note": ""}],\n'
            '  "catering": [{"item": "", "detail": "", "assumption": true, "assumption_note": ""}],\n'
            '  "transportation": [{"item": "", "detail": "", "assumption": true, "assumption_note": ""}],\n'
            '  "registration": [{"item": "", "detail": "", "assumption": true, "assumption_note": ""}],\n'
            '  "setup_checklist": [{"item": "", "detail": "", "assumption": false, "assumption_note": ""}],\n'
            '  "event_day_operations": [{"item": "", "detail": "", "assumption": false, "assumption_note": ""}]\n'
            "}\n"
            "Do not claim the venue capacity is confirmed unless the user stated it."
        )
        data = await self.client.complete_json(
            system_prompt=system,
            user_prompt=user,
            temperature=0.35,
        )
        return LogisticsOutput.model_validate(data)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class HiveOrchestrator:
    def __init__(self) -> None:
        self.client = GrokClient()
        self.manager = EventManagerAgent(self.client)
        self.planning = PlanningAgent(self.client)
        self.budget = BudgetAgent(self.client)
        self.logistics = LogisticsAgent(self.client)

    async def create_event(self, event: EventRequest) -> CreateEventResponse:
        workflow = [
            {"id": "understand", "label": "Event Manager — Understanding request", "status": "working"},
            {"id": "delegate", "label": "Event Manager — Delegating tasks", "status": "pending"},
            {"id": "planning", "label": "Planning Agent — Working", "status": "pending"},
            {"id": "budget", "label": "Budget Agent — Working", "status": "pending"},
            {"id": "logistics", "label": "Logistics Agent — Working", "status": "pending"},
            {"id": "validate", "label": "Event Manager — Collecting and validating results", "status": "pending"},
            {"id": "final", "label": "Final Event Plan — Ready", "status": "pending"},
        ]

        agents: dict[str, AgentResult] = {
            "event_manager": AgentResult(
                name="Event Manager",
                role="Orchestrator",
                status="working",
            ),
            "planning": AgentResult(name="Planning Agent", role=self.planning.role, status="pending"),
            "budget": AgentResult(name="Budget Agent", role=self.budget.role, status="pending"),
            "logistics": AgentResult(name="Logistics Agent", role=self.logistics.role, status="pending"),
        }
        warnings: list[str] = []
        errors: list[str] = []

        def mark(step_id: str, status: str) -> None:
            for step in workflow:
                if step["id"] == step_id:
                    step["status"] = status

        try:
            briefs = await self.manager.prepare_briefs(event)
        except Exception as exc:
            logger.exception("Event Manager briefing failed")
            message = _public_error(exc)
            errors.append(message)
            agents["event_manager"].status = "failed"
            agents["event_manager"].error = message
            mark("understand", "failed")
            return CreateEventResponse(
                success=False,
                event=as_plain(event),
                agents=agents,
                warnings=warnings,
                errors=errors,
                workflow=workflow,
            )

        mark("understand", "completed")
        mark("delegate", "completed")
        agents["event_manager"].output = {
            "understood_request": briefs.understood_request,
            "extracted_requirements": briefs.extracted_requirements,
            "briefs": {
                "planning": briefs.planning_brief,
                "budget": briefs.budget_brief,
                "logistics": briefs.logistics_brief,
            },
            "open_questions": briefs.open_questions,
        }
        agents["planning"].status = "working"
        agents["budget"].status = "working"
        agents["logistics"].status = "working"
        mark("planning", "working")
        mark("budget", "working")
        mark("logistics", "working")

        planning_task = asyncio.create_task(self.planning.run(event, briefs.planning_brief))
        budget_task = asyncio.create_task(self.budget.run(event, briefs.budget_brief))
        logistics_task = asyncio.create_task(self.logistics.run(event, briefs.logistics_brief))

        planning_raw, budget_raw, logistics_raw = await asyncio.gather(
            planning_task,
            budget_task,
            logistics_task,
            return_exceptions=True,
        )

        planning = _assign_agent(agents["planning"], planning_raw, PlanningOutput, "planning", mark, errors)
        budget = _assign_agent(agents["budget"], budget_raw, BudgetOutput, "budget", mark, errors)
        logistics = _assign_agent(agents["logistics"], logistics_raw, LogisticsOutput, "logistics", mark, errors)

        if planning is None or budget is None or logistics is None:
            agents["event_manager"].status = "failed"
            agents["event_manager"].error = (
                "An essential specialist agent failed, so a coordinated final plan was not produced."
            )
            mark("validate", "failed")
            mark("final", "failed")
            errors.append("Final Event Plan was not created because one or more essential agents failed.")
            return CreateEventResponse(
                success=False,
                event=as_plain(event),
                agents=agents,
                warnings=warnings,
                errors=errors,
                workflow=workflow,
            )

        mark("validate", "working")
        reconciliation = reconcile_budget(budget, event.budget, event.currency)
        agents["budget"].output = {
            **as_plain(budget),
            "reconciliation": as_plain(reconciliation),
        }
        if reconciliation.over_budget:
            warnings.append(
                f"Estimated allocations ({event.currency} {reconciliation.total_allocated:,.2f}) "
                f"exceed the available budget ({event.currency} {reconciliation.available_budget:,.2f})."
            )
        warnings.extend(budget.warnings)
        warnings.extend(briefs.open_questions)

        specialist_warnings = list(warnings)
        try:
            final_plan = await self.manager.compose_final_plan(
                event,
                briefs,
                planning,
                logistics,
                reconciliation,
                specialist_warnings,
            )
        except Exception as exc:
            logger.exception("Event Manager final plan failed")
            message = _public_error(exc)
            errors.append(message)
            agents["event_manager"].status = "failed"
            agents["event_manager"].error = message
            mark("validate", "failed")
            mark("final", "failed")
            return CreateEventResponse(
                success=False,
                event=as_plain(event),
                agents=agents,
                budget_reconciliation=reconciliation,
                warnings=warnings,
                errors=errors,
                workflow=workflow,
            )

        mark("validate", "completed")
        mark("final", "completed")
        agents["event_manager"].status = "completed"
        final_payload = {
            **as_plain(final_plan),
            "prioritized_tasks": as_plain(planning).get("prioritized_checklist", []),
            "budget_reconciliation": as_plain(reconciliation),
        }
        agents["event_manager"].output = {
            **(agents["event_manager"].output or {}),
            "final_plan": final_payload,
        }

        return CreateEventResponse(
            success=True,
            event=as_plain(event),
            agents=agents,
            final_plan=final_payload,
            budget_reconciliation=reconciliation,
            warnings=_unique(warnings + final_plan.conflicts + final_plan.unresolved_questions),
            errors=errors,
            workflow=workflow,
        )


def _assign_agent(
    result: AgentResult,
    raw: Any,
    model_type: type[BaseModel],
    step_id: str,
    mark,
    errors: list[str],
) -> BaseModel | None:
    if isinstance(raw, Exception):
        message = _public_error(raw)
        logger.warning("%s failed: %s", result.name, message)
        result.status = "failed"
        result.error = message
        mark(step_id, "failed")
        errors.append(f"{result.name} failed: {message}")
        return None
    try:
        parsed = raw if isinstance(raw, model_type) else model_type.model_validate(raw)
        result.status = "completed"
        result.output = as_plain(parsed)
        mark(step_id, "completed")
        return parsed
    except ValidationError as exc:
        message = f"{result.name} returned a structure that could not be validated."
        logger.warning("%s validation error: %s", result.name, exc)
        result.status = "failed"
        result.error = message
        mark(step_id, "failed")
        errors.append(message)
        return None


def _public_error(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "The AI request timed out. Retry the request."
    if isinstance(exc, RuntimeError):
        return str(exc)
    if isinstance(exc, ValidationError):
        return "An agent returned invalid structured data."
    logger.debug("Internal error detail: %s", traceback.format_exc())
    return "An unexpected error occurred while contacting the AI provider."


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = (item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "EventHive AI starting | model=%s | base=%s | key_configured=%s",
        GROK_MODEL,
        GROK_API_BASE,
        api_key_configured(),
    )
    yield


app = FastAPI(
    title="EventHive AI",
    description="Multi-agent event planning orchestrated by an Event Manager.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

orchestrator = HiveOrchestrator()


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "errors": ["The server encountered an unexpected error. Check backend logs."],
            "detail": "internal_error",
        },
    )


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "EventHive AI",
        "status": "ok",
        "health": "healthy",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    configured = api_key_configured()
    return {
        "status": "ok",
        "service": "EventHive AI",
        "ai_configured": configured,
        "model": GROK_MODEL if configured else None,
        "api_base_host": GROK_API_BASE.replace("https://", "").replace("http://", ""),
        "setup_message": None
        if configured
        else (
            "GROK_API_KEY is not configured. Add it to the backend .env file for local use, "
            "or set it as an environment variable on your host (Render). Never put the key in frontend code."
        ),
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/create-event", response_model=CreateEventResponse)
async def create_event(payload: EventRequest) -> CreateEventResponse:
    if not api_key_configured():
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "errors": [
                    "GROK_API_KEY is missing. Configure it in .env or your hosting environment, then retry."
                ],
            },
        )
    logger.info(
        "create-event | type=%s | attendees=%s | date=%s | budget=%s %s",
        payload.event_type,
        payload.attendees,
        payload.date,
        payload.currency,
        payload.budget,
    )
    try:
        result = await asyncio.wait_for(
            orchestrator.create_event(payload),
            timeout=ORCHESTRATION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        logger.warning("Orchestration timed out")
        raise HTTPException(
            status_code=504,
            detail={
                "success": False,
                "errors": ["The multi-agent workflow timed out. Retry without changing your form values."],
            },
        ) from exc

    if not result.success:
        logger.info("create-event completed with agent failures: %s", result.errors)
    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
