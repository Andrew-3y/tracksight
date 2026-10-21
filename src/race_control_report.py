"""Create a one-page, path-free PDF race-control review report."""

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


REPORT_DISCLAIMER = (
    "Review aid only. TrackSight does not establish fault, contact, a rule violation, "
    "or an incident outcome; it does not claim vehicle telemetry."
)


def build_race_control_report_pdf(run_name, run_summary, priority_cases, action_log, session_comparison=None):
    """Return an A4 report with compact, review-ready rather than punitive language."""
    stream = BytesIO()
    document = SimpleDocTemplate(
        stream, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm,
    )
    styles = getSampleStyleSheet()
    title = styles["Title"]
    title.fontSize = 18
    title.leading = 22
    body = styles["BodyText"]
    body.fontSize = 8.5
    body.leading = 11
    heading = styles["Heading2"]
    heading.fontSize = 11
    heading.leading = 14
    story = [
        Paragraph("TrackSight Marshal | Race-control review report", title),
        Paragraph(f"Completed run: <b>{run_name}</b>", body),
        Spacer(1, 4 * mm),
    ]
    kpis = [
        ["Analysed duration", f"{run_summary.get('processed_frame_count', 0) / run_summary.get('frames_per_second', 1):.1f} s" if run_summary.get("frames_per_second", 0) else "unknown"],
        ["Trusted ID segments", str(run_summary.get("trusted_track_segments", 0))],
        ["Review-now cases", str(sum(case["priority"] == "Review now" for case in priority_cases))],
        ["Recorded human actions", str(len(action_log.get("actions", [])))],
    ]
    kpi_table = Table(kpis, colWidths=[54 * mm, 36 * mm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF3F8")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9AA7B4")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([Paragraph("Run summary", heading), kpi_table, Spacer(1, 4 * mm), Paragraph("Priority review cases", heading)])
    rows = [["Priority", "Vehicles", "Video time", "Review support"]]
    for case in priority_cases[:3]:
        rows.append([
            case["priority"], case["track_ids_label"],
            f"{case['start_time_seconds']:.2f}-{case['end_time_seconds']:.2f} s",
            f"{case['event_count']} time-local evidence record(s)",
        ])
    if len(rows) == 1:
        rows.append(["No priority cases", "-", "-", "No quality-gated case was available."])
    case_table = Table(rows, colWidths=[36 * mm, 30 * mm, 35 * mm, 70 * mm], repeatRows=1)
    case_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D3557")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9AA7B4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([case_table, Spacer(1, 4 * mm)])
    if session_comparison and session_comparison.get("comparable"):
        story.append(Paragraph(
            "Session comparison: " + session_comparison["message"], body
        ))
        story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(REPORT_DISCLAIMER, body))
    document.build(story)
    return stream.getvalue()
