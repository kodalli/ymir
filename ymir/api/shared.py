"""Shared resources for routes."""

from fastapi import Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="ymir/templates")


def is_htmx_request(request: Request) -> bool:
    """Check if the request is from HTMX."""
    return request.headers.get("HX-Request") == "true"


def render_page(
    request: Request,
    template: str,
    context: dict,
    page_title: str = "Dashboard",
):
    """Render a page template, returning full page for direct access or fragment for HTMX."""
    context["request"] = request

    if is_htmx_request(request):
        # HTMX request - return just the fragment
        return templates.TemplateResponse(template, context)
    else:
        # Direct browser access - return full page with content
        context["page_template"] = template
        context["page_title"] = page_title
        return templates.TemplateResponse("_full_page.html", context)
