#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "solveit-client @ git+https://github.com/AnswerDotAI/solveit_client.git@0549138bad01c40a79079625332ee1c8611b1fa7",
#     "httpx",
#     "fastcore",
# ]
# ///

"""SolveIt dialog tools — AI-friendly CLI for reading and manipulating SolveIt dialogs."""

import argparse, json, sys, time
from urllib.parse import urlparse, parse_qs

from solveit_client.core import SolveItClient, Dialog, Message


# --- Helpers ---

def parse_url(url):
    """Extract (domain, dialog_name) from a SolveIt dialog URL."""
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}/"
    name = parse_qs(parsed.query).get('name', [None])[0]
    if not name:
        raise ValueError(f"No 'name' query parameter in URL: {url}")
    return domain, name


def get_dialog(url):
    """Open a Dialog from a SolveIt dialog URL."""
    domain, name = parse_url(url)
    return Dialog(name, SolveItClient(domain))


def msg_to_dict(m, index=None):
    """Convert a Message to a plain dict, omitting empty fields."""
    d = {"id": m.id, "type": m.msg_type}
    if index is not None:
        d["index"] = index
    d["content"] = m.content
    if m.output:
        d["output"] = m.output
    return d


# --- Tools ---

def read_dialog(args):
    dlg = get_dialog(args.url)
    messages = [msg_to_dict(m, index=i) for i, m in enumerate(dlg.messages)]
    return {"url": args.url, "name": dlg.name, "mode": dlg.mode, "messages": messages}


def add_message(args):
    dlg = get_dialog(args.url)
    # Map our friendly placement names to SolveIt's enum values
    placement_map = {"after": "add_after", "before": "add_before",
                     "at_end": "at_end", "at_start": "at_start"}
    kw = dict(dlg_name=dlg.name, content=args.content,
              msg_type=args.type, placement=placement_map[args.placement])
    if args.ref_id:
        kw["id_"] = args.ref_id
    mid = dlg.cli('/add_relative_', **kw)
    msg = Message(mid, dlg)
    return msg_to_dict(msg)


def exec_message(args):
    dlg = get_dialog(args.url)
    msg = Message(args.message_id, dlg)
    msg.exec(timeout=args.timeout)
    return msg_to_dict(msg)


def update_message(args):
    dlg = get_dialog(args.url)
    msg = Message(args.message_id, dlg)
    kw = {}
    if args.content is not None:
        kw["content"] = args.content
    if args.type is not None:
        kw["msg_type"] = args.type
    if not kw:
        raise ValueError("Nothing to update — provide --content or --type")
    msg.update(**kw)
    return msg_to_dict(msg)


def delete_message(args):
    dlg = get_dialog(args.url)
    msg = Message(args.message_id, dlg)
    msg.delete()
    return {"id": args.message_id, "deleted": True}


def run_dialog(args):
    dlg = get_dialog(args.url)
    # Snapshot outputs before execution
    before = {m.id: m.output for m in dlg.messages}
    dlg.run_all()
    # Poll until no messages are still running
    deadline = time.time() + args.timeout
    after_msgs = dlg.messages
    while time.time() < deadline:
        time.sleep(0.5)
        after_msgs = dlg.messages
        if not any(m.data.get('run') for m in after_msgs):
            break
    # Compare outputs to summarise what happened
    executed = skipped = errors = 0
    for m in after_msgs:
        if m.output != before.get(m.id):
            executed += 1
            if m.output and 'Traceback' in m.output:
                errors += 1
        else:
            skipped += 1
    return {"executed": executed, "skipped": skipped, "errors": errors}


# --- CLI ---

def main():
    p = argparse.ArgumentParser(
        prog="solveit_tools",
        description="AI-friendly CLI for SolveIt dialogs. All output is JSON.")
    sub = p.add_subparsers(dest="command", required=True)

    # read_dialog
    s = sub.add_parser("read_dialog", help="Read all messages in a dialog")
    s.add_argument("url", help="SolveIt dialog URL")

    # add_message
    s = sub.add_parser("add_message", help="Add a new message to a dialog")
    s.add_argument("url", help="SolveIt dialog URL")
    s.add_argument("content", help="Message content (source code, markdown, or prompt text)")
    s.add_argument("--type", default="code",
                   choices=["code", "note", "prompt", "raw"],
                   help="Message type (default: code)")
    s.add_argument("--placement", default="at_end",
                   choices=["at_end", "at_start", "after", "before"],
                   help="Where to place the message (default: at_end)")
    s.add_argument("--ref-id",
                   help="Reference message ID (required when placement is 'after' or 'before')")

    # exec_message
    s = sub.add_parser("exec_message", help="Execute a single message and return its output")
    s.add_argument("url", help="SolveIt dialog URL")
    s.add_argument("message_id", help="ID of the message to execute")
    s.add_argument("--timeout", type=float, default=60,
                   help="Max seconds to wait for execution (default: 60)")

    # update_message
    s = sub.add_parser("update_message", help="Update a message's content or type")
    s.add_argument("url", help="SolveIt dialog URL")
    s.add_argument("message_id", help="ID of the message to update")
    s.add_argument("--content", help="New content")
    s.add_argument("--type", choices=["code", "note", "prompt", "raw"],
                   help="New message type")

    # delete_message
    s = sub.add_parser("delete_message", help="Delete a message")
    s.add_argument("url", help="SolveIt dialog URL")
    s.add_argument("message_id", help="ID of the message to delete")

    # run_dialog
    s = sub.add_parser("run_dialog",
                       help="Execute the entire dialog (skips prompt cells by default)")
    s.add_argument("url", help="SolveIt dialog URL")
    s.add_argument("--timeout", type=float, default=120,
                   help="Max seconds to wait for completion (default: 120)")

    args = p.parse_args()
    handlers = dict(read_dialog=read_dialog, add_message=add_message,
                    exec_message=exec_message, update_message=update_message,
                    delete_message=delete_message, run_dialog=run_dialog)
    try:
        result = handlers[args.command](args)
        json.dump(result, sys.stdout, indent=2)
        print()
    except Exception as e:
        json.dump({"error": str(e)}, sys.stderr, indent=2)
        print(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
