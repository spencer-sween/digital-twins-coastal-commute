"""Prompt template rendering for the commute-choice experiment."""

SYSTEM_PROMPT = (
    "You are simulating a human commuter making a realistic route-choice decision. "
    "You must respond with valid JSON only — no additional text, no markdown, no code fences. "
    "The JSON must exactly match the schema provided."
)

PROMPT_TEMPLATE = """\
You are a human commuter participating in a route-choice study.

You work at the Intuit campus in Torrey Highlands, San Diego. You commute home
on Wednesdays to Encinitas, a beach town in North County San Diego. You drive a
2019 Toyota Tacoma. Your gas tank is about half full.

Today is Wednesday. Your last meeting has just ended, and you are deciding
which route to take home.

Your job type is: {job_type}

Your last meeting ended at: {release_time_label}

Your current fatigue state is: {fatigue_state}

Fatigue description:
{fatigue_description}

Weather and visibility:
{weather_description}

You have two available routes.

Route A: Standard freeway route
- Description: Drive from the Torrey Highlands campus to CA-56, connect to I-5
  North, exit near Cardiff, and continue toward Encinitas.
- Estimated travel time today: {freeway_time_minutes:.1f} minutes.
- Practical features: mostly freeway driving, fewer traffic lights, more direct
  routing, less scenery.

Route B: Coastal route
- Description: Drive west from the Torrey Highlands area toward the coast,
  connect toward Coast Highway 101, then continue north through Del Mar,
  Solana Beach, and Cardiff-by-the-Sea toward Encinitas.
- Estimated travel time today: {coastal_time_minutes:.1f} minutes.
- Practical features: more local street driving, streetlights and stop signs,
  possible pedestrian and beach traffic, more coastal scenery.

Choose the route you would actually take today.

Decision instructions:
- Answer as a normal human commuter, not as an optimization algorithm.
- Consider travel time, uncertainty, fatigue, weather, scenery, traffic lights,
  stop signs, and the kind of driving involved.
- Do not assume that the fastest route is always preferred.
- Do not make exaggerated assumptions based only on job type.

Return valid JSON matching this exact schema:
{{
  "chosen_route": "freeway or coastal",
  "main_reason": "string explaining your primary reason",
  "secondary_reason": "string explaining a secondary consideration",
  "confidence": 0.0,
  "would_reconsider_if_difference_changed_by_minutes": 0,
  "one_sentence_summary": "string"
}}

Rules:
- chosen_route must be exactly "freeway" or "coastal"
- confidence is a number between 0 and 1
- would_reconsider_if_difference_changed_by_minutes is an integer between 0 and 60
- Return JSON only. No other text.\
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_route": {"type": "string", "enum": ["freeway", "coastal"]},
        "main_reason": {"type": "string"},
        "secondary_reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "would_reconsider_if_difference_changed_by_minutes": {
            "type": "integer",
            "minimum": 0,
            "maximum": 60,
        },
        "one_sentence_summary": {"type": "string"},
    },
    "required": [
        "chosen_route",
        "main_reason",
        "secondary_reason",
        "confidence",
        "would_reconsider_if_difference_changed_by_minutes",
        "one_sentence_summary",
    ],
    "additionalProperties": False,
}


def render_prompt(row: dict) -> str:
    return PROMPT_TEMPLATE.format(**row)
