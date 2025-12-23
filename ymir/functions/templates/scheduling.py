"""Scheduling domain function templates.

These match the tool schemas used in the llm-agent-training benchmark.
"""

from ymir.functions.schemas import FunctionDefinition, ScenarioTemplate

SCHEDULING_FUNCTIONS = [
    FunctionDefinition(
        name="search_patient",
        description="Search for a patient by first and/or last name. Returns a list of matching patients with their IDs.",
        parameters={
            "type": "object",
            "properties": {
                "first_name": {
                    "type": "string",
                    "description": "Patient's first name",
                },
                "last_name": {
                    "type": "string",
                    "description": "Patient's last name",
                },
            },
            "required": [],
        },
        category="scheduling",
    ),
    FunctionDefinition(
        name="get_patient_info",
        description="Get detailed information about a specific patient by their ID.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The unique patient identifier",
                },
            },
            "required": ["patient_id"],
        },
        category="scheduling",
    ),
    FunctionDefinition(
        name="get_open_slots",
        description="Get available appointment slots for a specific date and optionally a specific provider.",
        parameters={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format",
                },
                "provider_id": {
                    "type": "string",
                    "description": "Optional provider/doctor ID to filter slots",
                },
            },
            "required": ["date"],
        },
        category="scheduling",
    ),
    FunctionDefinition(
        name="get_providers",
        description="List available healthcare providers/doctors, optionally filtered by specialty.",
        parameters={
            "type": "object",
            "properties": {
                "specialty": {
                    "type": "string",
                    "description": "Medical specialty to filter by (e.g., 'cardiology', 'general')",
                },
            },
            "required": [],
        },
        category="scheduling",
    ),
    FunctionDefinition(
        name="book_appointment",
        description="Book an appointment for a patient at a specific slot.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The patient's unique identifier",
                },
                "slot_id": {
                    "type": "string",
                    "description": "The slot identifier to book",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the appointment",
                },
            },
            "required": ["patient_id", "slot_id"],
        },
        category="scheduling",
    ),
    FunctionDefinition(
        name="cancel_appointment",
        description="Cancel an existing appointment.",
        parameters={
            "type": "object",
            "properties": {
                "appointment_id": {
                    "type": "string",
                    "description": "The appointment ID to cancel",
                },
            },
            "required": ["appointment_id"],
        },
        category="scheduling",
    ),
    FunctionDefinition(
        name="get_patient_appointments",
        description="Get all appointments for a specific patient.",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The patient's unique identifier",
                },
                "include_past": {
                    "type": "boolean",
                    "description": "Whether to include past appointments",
                },
            },
            "required": ["patient_id"],
        },
        category="scheduling",
    ),
]

SCHEDULING_SCENARIO = ScenarioTemplate(
    id="medical-scheduling",
    name="Medical Appointment Scheduling",
    description="A scheduling assistant that helps patients book, manage, and cancel medical appointments.",
    category="scheduling",
    functions=SCHEDULING_FUNCTIONS,
    system_prompt="""You are a helpful medical scheduling assistant. Your role is to help patients:
- Find and book appointments with healthcare providers
- Check available appointment slots
- Manage existing appointments (view, cancel, reschedule)
- Look up patient information when needed

Always be professional, courteous, and efficient. When booking appointments, confirm all details with the patient before finalizing.

To perform actions, use the available tools by responding with a tool call in this format:
<tool_call>{"name": "tool_name", "arguments": {"arg1": "value1"}}</tool_call>

After receiving tool results, continue helping the patient or provide your final response.""",
    example_queries=[
        "I need to schedule an appointment with Dr. Smith for next week",
        "Can you check if there are any openings tomorrow afternoon?",
        "I'd like to cancel my appointment on Friday",
        "What appointments do I have scheduled?",
        "I need to see a cardiologist as soon as possible",
        "Can you book me for a checkup with any available doctor?",
    ],
    mock_responses={
        "search_patient": {
            "patients": [
                {"id": "P001", "first_name": "John", "last_name": "Smith", "dob": "1985-03-15"},
            ]
        },
        "get_open_slots": {
            "slots": [
                {"id": "S001", "time": "09:00 AM", "provider": "Dr. Johnson"},
                {"id": "S002", "time": "02:00 PM", "provider": "Dr. Johnson"},
                {"id": "S003", "time": "04:30 PM", "provider": "Dr. Williams"},
            ]
        },
        "book_appointment": {
            "success": True,
            "appointment_id": "A001",
            "message": "Appointment booked successfully",
        },
    },
)
