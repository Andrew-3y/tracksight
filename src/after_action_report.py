"""Create a concise, path-free PDF handoff for one selected review case."""

from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


AFTER_ACTION_DISCLAIMER = (
    "Review aid only. This report records what TrackSight observed and what a human reviewer decided; "
    "it does not establish contact, fault, a rule breach, or an incident outcome."
)


def build_after_action_report_pdf(run_name, case, ladder, correlation, case_rows):
    """Build a compact report that stays within the selected case's saved evidence."""
    stream = BytesIO()
    document = SimpleDocTemplate(
        stream, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm,
    )
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontSize = 8.5
    body.leading = 11
    heading = styles["Heading2"]
    heading.fontSize = 11
    title = styles["Title"]
    story = [
        Paragraph("TrackSight Marshal | After-action review", title),
        Paragraph(f"Completed run: <b>{escape(str(run_name))}</b>", body),
        Spacer(1, 3 * mm),
        Paragraph("Decision ladder", heading),
        Paragraph(
            f"<b>{escape(ladder['status'])}</b> — {escape(ladder['recommended_action'])}",
            body,
        ),
        Spacer(1, 3 * mm),
    ]
    case_table = Table([
        ["Vehicles", ", ".join(map(str, case["track_ids"]))],
        ["Video time", f"{case['start_time_seconds']:.2f}–{case['end_time_seconds']:.2f} s"],
        ["Saved evidence records", str(len(case_rows))],
        ["Evidence sources", ", ".join(ladder["source_evidence_types"]) or "none"],
    ], colWidths=[48 * mm, 120 * mm])
    case_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3F8")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9AA7B4")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([case_table, Spacer(1, 4 * mm), Paragraph("Evidence availability", heading)])
    rows = [["Source", "Status", "Detail"]] + [
        [item["source"], item["status"], item["detail"]] for item in correlation
    ]
    evidence_table = Table(rows, colWidths=[45 * mm, 30 * mm, 93 * mm], repeatRows=1)
    evidence_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D3557")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9AA7B4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([evidence_table, Spacer(1, 4 * mm), Paragraph(AFTER_ACTION_DISCLAIMER, body)])
    document.build(story)
    return stream.getvalue()
