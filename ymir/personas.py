"""Persona presets for user simulation."""

from pydantic import BaseModel, Field


class PersonaPreset(BaseModel):
    """A preset persona for simulated user interactions."""

    id: str
    name: str
    icon: str  # Lucide icon name
    background: str
    goal: str
    tags: list[str] = Field(default_factory=list)
    category: str = "general"  # Links to scenario category


# Medical Scheduling Persona Presets
SCHEDULING_PERSONAS: list[PersonaPreset] = [
    PersonaPreset(
        id="elderly-patient",
        name="Elderly Patient",
        icon="heart-pulse",
        background="72-year-old retired teacher. Has trouble hearing on the phone. Takes multiple medications and sees several specialists. Prefers morning appointments because medications make them tired in the afternoon.",
        goal="Schedule a follow-up appointment with their cardiologist within the next two weeks. Also wants to check if they have any other upcoming appointments.",
        tags=["patient", "complex-needs", "morning-preference"],
        category="scheduling",
    ),
    PersonaPreset(
        id="busy-professional",
        name="Busy Professional",
        icon="briefcase",
        background="35-year-old marketing executive. Very limited availability - only free during lunch breaks (12-1pm) or after 5pm. Impatient with slow processes. Prefers efficiency over small talk.",
        goal="Book any available appointment this week for a routine checkup. Must be during lunch or after work hours.",
        tags=["time-constrained", "efficient", "flexible-provider"],
        category="scheduling",
    ),
    PersonaPreset(
        id="anxious-first-timer",
        name="Anxious First-Timer",
        icon="help-circle",
        background="28-year-old new to the city. First time using this medical system. Nervous about medical visits in general. Has many questions about the process.",
        goal="Schedule a new patient appointment. Wants to understand the registration process and what to bring.",
        tags=["new-patient", "anxious", "information-seeking"],
        category="scheduling",
    ),
    PersonaPreset(
        id="frustrated-patient",
        name="Frustrated Patient",
        icon="frown",
        background="45-year-old with chronic back pain. Their last two appointments were cancelled by the clinic with short notice. Feeling upset and considering switching providers.",
        goal="Reschedule the cancelled appointment and express frustration. Wants assurance this won't happen again.",
        tags=["upset", "rescheduling", "retention-risk"],
        category="scheduling",
    ),
    PersonaPreset(
        id="parent-booking",
        name="Parent Booking for Child",
        icon="baby",
        background="Parent of a 6-year-old child who needs a vaccination appointment. Juggling their own work schedule with the child's school schedule. Prefers appointments that don't require taking the child out of school.",
        goal="Book a vaccination appointment for their child, preferably after school hours (3:30pm+) or on a weekend if available.",
        tags=["pediatric", "scheduling-constraints", "parent"],
        category="scheduling",
    ),
]


def get_personas_for_category(category: str) -> list[PersonaPreset]:
    """Get persona presets for a specific scenario category."""
    if category == "scheduling":
        return SCHEDULING_PERSONAS
    # Add more category mappings as needed
    return []


def get_persona_by_id(persona_id: str) -> PersonaPreset | None:
    """Get a specific persona by ID."""
    all_personas = SCHEDULING_PERSONAS  # Add more as categories grow
    for persona in all_personas:
        if persona.id == persona_id:
            return persona
    return None
