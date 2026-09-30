"""Command-line entry point: transcribe a local file and print or save the result.

    python transcribe.py samples/sample.mp3
    python transcribe.py call.wav --split-channels --format srt -o call.srt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from app.audio import InvalidAudioError
from app.formats import to_srt, to_vtt
from app.pipeline import Options, Pipeline
from app.transcriber import get_transcriber


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with segment timestamps.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--language", help="ISO code, e.g. en. Auto-detected if omitted.")
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--split-channels", action="store_true", help="stereo calls: one speaker per channel")
    parser.add_argument("--format", choices=["json", "srt", "vtt"], default="json")
    parser.add_argument("-o", "--output", type=Path, help="write to a file instead of stdout")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    options = Options(language=args.language, word_timestamps=args.word_timestamps,
                      split_channels=args.split_channels)

    try:
        result = Pipeline(get_transcriber()).run(
            args.file, options,
            progress=lambda done, total: print(f"chunk {done}/{total}", file=sys.stderr),
        )
    except InvalidAudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.format == "srt":
        out = to_srt(result)
    elif args.format == "vtt":
        out = to_vtt(result)
    else:
        out = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(out)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
