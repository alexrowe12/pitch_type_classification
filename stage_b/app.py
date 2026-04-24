#!/usr/bin/env python3
"""
Local Streamlit app for Stage B release/catch review.

Usage:
    streamlit run stage_b/app.py
"""

import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage_b.paths import (
    FRAME_EXPORTS_CSV,
    MANUAL_EVENTS_CSV,
    WEAK_EVENTS_CSV,
    ensure_stage_b_dirs,
)


EVENT_FIELDNAMES = [
    "clip_id",
    "pitch_type",
    "release_frame_idx",
    "catch_frame_idx",
    "usable",
    "weak_release_frame_idx",
    "weak_catch_frame_idx",
    "weak_confidence",
    "weak_reason",
    "notes",
    "labeled_at_utc",
]


def load_csv_rows(path: Path) -> list[dict]:
    """Load CSV rows if present, else return an empty list."""
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def group_frame_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """Group exported candidate frame rows by clip."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row["frame_idx"] = int(row["frame_idx"])
        grouped[row["clip_id"]].append(row)
    for clip_rows in grouped.values():
        clip_rows.sort(key=lambda row: row["frame_idx"])
    return grouped


def load_weak_events() -> dict[str, dict]:
    """Load weak event guesses keyed by clip id."""
    events = {}
    for row in load_csv_rows(WEAK_EVENTS_CSV):
        row["release_frame_idx"] = int(row["release_frame_idx"])
        row["catch_frame_idx"] = int(row["catch_frame_idx"])
        row["confidence"] = float(row["confidence"])
        events[row["clip_id"]] = row
    return events


def load_manual_events() -> list[dict]:
    """Load any saved manual Stage B events."""
    return load_csv_rows(MANUAL_EVENTS_CSV)


def save_manual_event(
    clip_rows: list[dict],
    weak_event: dict | None,
    release_frame_idx: int,
    catch_frame_idx: int,
    usable: bool,
    notes: str,
) -> None:
    """Append one manual event row."""
    first = clip_rows[0]
    MANUAL_EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = MANUAL_EVENTS_CSV.exists()
    with open(MANUAL_EVENTS_CSV, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "clip_id": first["clip_id"],
                "pitch_type": first["pitch_type"],
                "release_frame_idx": release_frame_idx,
                "catch_frame_idx": catch_frame_idx,
                "usable": "1" if usable else "0",
                "weak_release_frame_idx": weak_event["release_frame_idx"] if weak_event else "",
                "weak_catch_frame_idx": weak_event["catch_frame_idx"] if weak_event else "",
                "weak_confidence": f"{weak_event['confidence']:.6f}" if weak_event else "",
                "weak_reason": weak_event["reason"] if weak_event else "",
                "notes": notes,
                "labeled_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )


def remove_last_manual_event() -> bool:
    """Remove the most recently appended manual event."""
    rows = load_manual_events()
    if not rows:
        return False

    with open(MANUAL_EVENTS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows[:-1])
    return True


def get_next_clip_id(clip_ids: list[str], manual_rows: list[dict]) -> str | None:
    """Return the next unlabeled clip id."""
    labeled_clip_ids = {row["clip_id"] for row in manual_rows}
    for clip_id in clip_ids:
        if clip_id not in labeled_clip_ids:
            return clip_id
    return None


def nearest_available_frame(frame_indices: list[int], target: int) -> int:
    """Return the exported frame nearest to a target frame number."""
    return min(frame_indices, key=lambda frame_idx: abs(frame_idx - target))


def move_frame(frame_indices: list[int], current: int, delta: int) -> int:
    """Move a selected frame by index position within exported frames."""
    if current not in frame_indices:
        current = nearest_available_frame(frame_indices, current)
    current_index = frame_indices.index(current)
    next_index = max(0, min(len(frame_indices) - 1, current_index + delta))
    return frame_indices[next_index]


def clamp_event_order(frame_indices: list[int], release_frame_idx: int, catch_frame_idx: int) -> tuple[int, int]:
    """Keep release at or before catch using available exported frames."""
    release_frame_idx = nearest_available_frame(frame_indices, release_frame_idx)
    catch_frame_idx = nearest_available_frame(frame_indices, catch_frame_idx)
    if release_frame_idx > catch_frame_idx:
        catch_frame_idx = release_frame_idx
    return release_frame_idx, catch_frame_idx


def sample_display_rows(rows: list[dict], release_frame_idx: int, catch_frame_idx: int, max_frames: int) -> list[dict]:
    """Sample rows for display while forcing release/catch rows to be visible."""
    if len(rows) <= max_frames:
        return rows

    frame_to_row = {row["frame_idx"]: row for row in rows}
    required = {
        nearest_available_frame(list(frame_to_row), release_frame_idx),
        nearest_available_frame(list(frame_to_row), catch_frame_idx),
    }

    last_index = len(rows) - 1
    selected_indices = set()
    for output_index in range(max_frames):
        selected_indices.add(round(output_index * last_index / (max_frames - 1)))
    for frame_idx in required:
        selected_indices.add(rows.index(frame_to_row[frame_idx]))

    return [rows[index] for index in sorted(selected_indices)]


def local_rows(rows: list[dict], center_frame_idx: int, radius: int = 4) -> list[dict]:
    """Return a small local window of rows around one frame."""
    frame_indices = [row["frame_idx"] for row in rows]
    center_frame_idx = nearest_available_frame(frame_indices, center_frame_idx)
    center_index = frame_indices.index(center_frame_idx)
    start = max(0, center_index - radius)
    end = min(len(rows), center_index + radius + 1)
    return rows[start:end]


def ensure_state_for_clip(clip_id: str, clip_rows: list[dict], weak_event: dict | None) -> None:
    """Initialize Streamlit session state for the active clip."""
    import streamlit as st

    if st.session_state.get("stage_b_clip_id") == clip_id:
        return

    frame_indices = [row["frame_idx"] for row in clip_rows]
    if weak_event:
        release = nearest_available_frame(frame_indices, weak_event["release_frame_idx"])
        catch = nearest_available_frame(frame_indices, weak_event["catch_frame_idx"])
    else:
        release = frame_indices[len(frame_indices) // 3]
        catch = frame_indices[min(len(frame_indices) - 1, len(frame_indices) // 3 + 18)]

    if catch <= release:
        catch = frame_indices[min(len(frame_indices) - 1, frame_indices.index(release) + 12)]

    st.session_state.stage_b_clip_id = clip_id
    st.session_state.release_frame_idx = release
    st.session_state.catch_frame_idx = catch
    st.session_state.notes = ""
    st.session_state.jump_size = 5


def inject_hotkeys() -> None:
    """Inject keyboard shortcuts for the app controls."""
    import streamlit.components.v1 as components

    components.html(
        """
        <script>
        const doc = window.parent.document;
        if (!doc.__stageBHotkeysBound) {
          doc.__stageBHotkeysBound = true;
          doc.addEventListener("keydown", function(event) {
            const activeTag = doc.activeElement ? doc.activeElement.tagName : "";
            if (["INPUT", "TEXTAREA"].includes(activeTag)) {
              return;
            }
            const key = event.key.toLowerCase();
            const buttons = Array.from(doc.querySelectorAll("button"));
            const clickByText = (text) => {
              const button = buttons.find(btn => btn.innerText.trim() === text);
              if (button) {
                event.preventDefault();
                button.click();
              }
            };
            if (key === "a") {
              clickByText("Release -1 (A)");
            } else if (key === "d") {
              clickByText("Release +1 (D)");
            } else if (key === "j") {
              clickByText("Catch -1 (J)");
            } else if (key === "l") {
              clickByText("Catch +1 (L)");
            } else if (key === "enter") {
              clickByText("Save Usable (Enter)");
            } else if (key === "u") {
              clickByText("Mark Unusable (U)");
            } else if (key === "z") {
              clickByText("Undo (Z)");
            } else if (key === "s") {
              clickByText("Release -Jump (S)");
            } else if (key === "f") {
              clickByText("Release +Jump (F)");
            } else if (key === "k") {
              clickByText("Catch -Jump (K)");
            } else if (key === ";") {
              clickByText("Catch +Jump (;)");
            }
          });
        }
        </script>
        """,
        height=0,
    )


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Streamlit is not installed. Install it with `pip install streamlit`.") from exc

    ensure_stage_b_dirs()
    st.set_page_config(page_title="Stage B Release/Catch Review", layout="wide")
    st.title("Stage B Review")
    st.caption("Shortcuts: A/D release, J/L catch, S/F release jump, K/; catch jump, Enter save usable, U unusable, Z undo")

    frame_rows = load_csv_rows(FRAME_EXPORTS_CSV)
    if not frame_rows:
        st.warning("No Stage B candidate frames found. Run `python -m stage_b.export_candidates` first.")
        return

    grouped_rows = group_frame_rows(frame_rows)
    weak_events = load_weak_events()
    manual_rows = load_manual_events()
    clip_ids = sorted(grouped_rows)
    next_clip_id = get_next_clip_id(clip_ids, manual_rows)

    top_cols = st.columns([1, 1, 4])
    if top_cols[0].button("Undo (Z)", use_container_width=True, disabled=not manual_rows):
        if remove_last_manual_event():
            st.rerun()
    top_cols[1].metric("Labeled", f"{len({row['clip_id'] for row in manual_rows})}/{len(clip_ids)}")

    if next_clip_id is None:
        st.success("Stage B review complete.")
        return

    clip_rows = grouped_rows[next_clip_id]
    weak_event = weak_events.get(next_clip_id)
    ensure_state_for_clip(next_clip_id, clip_rows, weak_event)
    frame_indices = [row["frame_idx"] for row in clip_rows]

    st.subheader(f"{next_clip_id} | {clip_rows[0]['pitch_type']}")
    meta_cols = st.columns(5)
    meta_cols[0].metric("Frames", len(clip_rows))
    meta_cols[1].metric("Release", st.session_state.release_frame_idx)
    meta_cols[2].metric("Catch", st.session_state.catch_frame_idx)
    meta_cols[3].metric("Gap", st.session_state.catch_frame_idx - st.session_state.release_frame_idx)
    meta_cols[4].metric("Weak Conf", f"{weak_event['confidence']:.3f}" if weak_event else "n/a")
    if weak_event:
        st.caption(
            f"Weak guess: release={weak_event['release_frame_idx']}, "
            f"catch={weak_event['catch_frame_idx']}, reason={weak_event['reason']}"
        )

    control_top = st.columns([1.2, 1.2, 1.2, 2.4])
    jump_size = control_top[0].selectbox("Jump", [1, 3, 5, 10, 15], index=[1, 3, 5, 10, 15].index(st.session_state.get("jump_size", 5)))
    st.session_state.jump_size = jump_size
    release_frame_idx, catch_frame_idx = clamp_event_order(
        frame_indices,
        st.session_state.release_frame_idx,
        st.session_state.catch_frame_idx,
    )
    st.session_state.release_frame_idx = control_top[1].select_slider(
        "Release Frame",
        options=frame_indices,
        value=release_frame_idx,
    )
    st.session_state.catch_frame_idx = control_top[2].select_slider(
        "Catch Frame",
        options=frame_indices,
        value=max(st.session_state.release_frame_idx, catch_frame_idx),
    )
    st.session_state.release_frame_idx, st.session_state.catch_frame_idx = clamp_event_order(
        frame_indices,
        st.session_state.release_frame_idx,
        st.session_state.catch_frame_idx,
    )

    control_cols = st.columns(8)
    if control_cols[0].button("Release -1 (A)", use_container_width=True):
        st.session_state.release_frame_idx = move_frame(frame_indices, st.session_state.release_frame_idx, -1)
        st.rerun()
    if control_cols[1].button("Release +1 (D)", use_container_width=True):
        st.session_state.release_frame_idx = move_frame(frame_indices, st.session_state.release_frame_idx, 1)
        st.rerun()
    if control_cols[2].button("Release -Jump (S)", use_container_width=True):
        st.session_state.release_frame_idx = move_frame(frame_indices, st.session_state.release_frame_idx, -jump_size)
        st.rerun()
    if control_cols[3].button("Release +Jump (F)", use_container_width=True):
        st.session_state.release_frame_idx = move_frame(frame_indices, st.session_state.release_frame_idx, jump_size)
        st.rerun()
    if control_cols[4].button("Catch -1 (J)", use_container_width=True):
        st.session_state.catch_frame_idx = move_frame(frame_indices, st.session_state.catch_frame_idx, -1)
        st.rerun()
    if control_cols[5].button("Catch +1 (L)", use_container_width=True):
        st.session_state.catch_frame_idx = move_frame(frame_indices, st.session_state.catch_frame_idx, 1)
        st.rerun()
    if control_cols[6].button("Catch -Jump (K)", use_container_width=True):
        st.session_state.catch_frame_idx = move_frame(frame_indices, st.session_state.catch_frame_idx, -jump_size)
        st.rerun()
    if control_cols[7].button("Catch +Jump (;)", use_container_width=True):
        st.session_state.catch_frame_idx = move_frame(frame_indices, st.session_state.catch_frame_idx, jump_size)
        st.rerun()

    utility_cols = st.columns(3)
    if utility_cols[0].button("Set Catch = Release", use_container_width=True):
        st.session_state.catch_frame_idx = st.session_state.release_frame_idx
        st.rerun()
    if utility_cols[1].button("Reset Weak", use_container_width=True):
        st.session_state.stage_b_clip_id = None
        st.rerun()
    utility_cols[2].caption(f"Selected window: {st.session_state.release_frame_idx} -> {st.session_state.catch_frame_idx}")

    notes = st.text_input("Notes", key="notes")

    zoom_cols = st.columns(2)
    zoom_cols[0].markdown("**Release Zoom**")
    zoom_cols[1].markdown("**Catch Zoom**")
    release_zoom = local_rows(clip_rows, st.session_state.release_frame_idx, radius=4)
    catch_zoom = local_rows(clip_rows, st.session_state.catch_frame_idx, radius=4)
    for container, zoom_rows, target_frame, label in [
        (zoom_cols[0], release_zoom, st.session_state.release_frame_idx, "RELEASE"),
        (zoom_cols[1], catch_zoom, st.session_state.catch_frame_idx, "CATCH"),
    ]:
        cols = container.columns(len(zoom_rows))
        for col, row in zip(cols, zoom_rows):
            frame_idx = row["frame_idx"]
            title = f"**{label}**" if frame_idx == target_frame else "&nbsp;"
            col.markdown(title)
            col.image(row["frame_path"], caption=f"{frame_idx}", use_container_width=True)

    display_rows = sample_display_rows(
        clip_rows,
        release_frame_idx=st.session_state.release_frame_idx,
        catch_frame_idx=st.session_state.catch_frame_idx,
        max_frames=36,
    )
    cols = st.columns(6)
    for index, row in enumerate(display_rows):
        frame_idx = row["frame_idx"]
        col = cols[index % 6]
        button_key_prefix = f"{next_clip_id}_{frame_idx}"
        if frame_idx == st.session_state.release_frame_idx:
            col.markdown("**RELEASE**")
        elif frame_idx == st.session_state.catch_frame_idx:
            col.markdown("**CATCH**")
        elif st.session_state.release_frame_idx < frame_idx < st.session_state.catch_frame_idx:
            col.markdown("sequence")
        else:
            col.markdown("&nbsp;")
        col.image(row["frame_path"], caption=f"frame {frame_idx}", use_container_width=True)
        pick_cols = col.columns(2)
        if pick_cols[0].button("Set R", key=f"{button_key_prefix}_r", use_container_width=True):
            st.session_state.release_frame_idx = frame_idx
            st.rerun()
        if pick_cols[1].button("Set C", key=f"{button_key_prefix}_c", use_container_width=True):
            st.session_state.catch_frame_idx = frame_idx
            st.rerun()

    save_cols = st.columns(2)
    if save_cols[0].button("Save Usable (Enter)", use_container_width=True):
        save_manual_event(
            clip_rows=clip_rows,
            weak_event=weak_event,
            release_frame_idx=st.session_state.release_frame_idx,
            catch_frame_idx=st.session_state.catch_frame_idx,
            usable=True,
            notes=notes,
        )
        st.rerun()
    if save_cols[1].button("Mark Unusable (U)", use_container_width=True):
        save_manual_event(
            clip_rows=clip_rows,
            weak_event=weak_event,
            release_frame_idx=st.session_state.release_frame_idx,
            catch_frame_idx=st.session_state.catch_frame_idx,
            usable=False,
            notes=notes,
        )
        st.rerun()

    inject_hotkeys()


if __name__ == "__main__":
    main()
