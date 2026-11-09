#!/usr/bin/env python3
# Author: Yuxiang Huang
"Convert bounding boxes to integer-floor center coordinates."


import json
import csv
import argparse
import sys
from pathlib import Path


def calculate_centers_and_save(input_file_path, output_file_path, verbose=False):


    input_path = Path(input_file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")


    try:
        with open(input_file_path, 'r') as input_file:
            data = json.load(input_file)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in input file: {e}", e.doc, e.pos)


    if not isinstance(data, list):
        raise ValueError(f"Expected list at root level, got {type(data).__name__}")


    new_data = []
    num_converted = 0
    num_skipped = 0

    for idx, entry in enumerate(data):
        try:

            if not isinstance(entry, dict):
                if verbose:
                    (f'{idx}', sys.stderr)
                num_skipped += 1
                continue

            if 'name' not in entry:
                if verbose:
                    (f'{idx}', sys.stderr)
                num_skipped += 1
                continue

            if 'bounds' not in entry:
                if verbose:
                    (f'{idx}', sys.stderr)
                num_skipped += 1
                continue

            name = entry['name']
            bounds = entry['bounds']


            if not isinstance(bounds, list) or len(bounds) != 2:
                if verbose:
                    (f'{idx}', f'{name}', sys.stderr)
                num_skipped += 1
                continue

            if not isinstance(bounds[0], list) or not isinstance(bounds[1], list):
                if verbose:
                    (f'{idx}', f'{name}', sys.stderr)
                num_skipped += 1
                continue

            if len(bounds[0]) != len(bounds[1]):
                if verbose:
                    (f'{idx}', f'{name}', sys.stderr)
                num_skipped += 1
                continue


            center = [(bounds[0][i] + bounds[1][i]) // 2 for i in range(len(bounds[0]))]


            new_data.append({'name': name, 'bounds': center})
            num_converted += 1

        except Exception as e:
            if verbose:
                (f'{idx}', f'{e}', sys.stderr)
            num_skipped += 1
            continue


    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)


    with open(output_file_path, 'w') as output_file:
        json.dump(new_data, output_file, indent=4)


    stats = {
        'num_entries': len(data),
        'num_converted': num_converted,
        'num_skipped': num_skipped
    }

    return stats


def process_csv_batch(csv_file_path, verbose=False):

    csv_path = Path(csv_file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")


    file_pairs = []
    try:
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            first_row = True

            for row_num, row in enumerate(reader, start=1):
                if len(row) < 2:
                    (f'{row_num}', sys.stderr)
                    continue

                input_path = row[0].strip()
                output_path = row[1].strip()

                if not input_path or not output_path:
                    (f'{row_num}', sys.stderr)
                    continue


                if first_row:
                    first_row = False


                    is_likely_header = (
                        not input_path.startswith(('./', '/', '~', '../')) and
                        '.' not in Path(input_path).name and
                        any(keyword in input_path.lower() for keyword in ['input', 'source', 'from', 'path', 'file'])
                    )
                    if is_likely_header:
                        if verbose:
                            (f'{input_path}', f'{output_path}', sys.stderr)
                        continue


                file_pairs.append((input_path, output_path))

    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    if not file_pairs:
        raise ValueError("No valid file pairs found in CSV")


    batch_stats = {
        'total_files': len(file_pairs),
        'successful': 0,
        'failed': 0,
        'total_entries': 0,
        'total_converted': 0,
        'total_skipped': 0,
        'failed_files': []
    }


    print(f"\nProcessing {len(file_pairs)} file pairs from CSV...\n")

    for idx, (input_path, output_path) in enumerate(file_pairs, start=1):
        try:

            print(f"[{idx}/{len(file_pairs)}] Processing: {Path(input_path).name}")


            stats = calculate_centers_and_save(input_path, output_path, verbose=verbose)


            batch_stats['successful'] += 1
            batch_stats['total_entries'] += stats['num_entries']
            batch_stats['total_converted'] += stats['num_converted']
            batch_stats['total_skipped'] += stats['num_skipped']


            print(f"  Converted: {stats['num_converted']}/{stats['num_entries']} neurons")
            if stats['num_skipped'] > 0:
                (f"{stats['num_skipped']}",)

        except FileNotFoundError as e:
            batch_stats['failed'] += 1
            error_msg = f"File not found: {e}"
            batch_stats['failed_files'].append((input_path, error_msg))
            print(f"  Error: {error_msg}", file=sys.stderr)

        except json.JSONDecodeError as e:
            batch_stats['failed'] += 1
            error_msg = f"Invalid JSON: {e}"
            batch_stats['failed_files'].append((input_path, error_msg))
            print(f"  Error: {error_msg}", file=sys.stderr)

        except Exception as e:
            batch_stats['failed'] += 1
            error_msg = str(e)
            batch_stats['failed_files'].append((input_path, error_msg))
            print(f"  Error: {error_msg}", file=sys.stderr)

        print()

    return batch_stats


def main():

    parser = argparse.ArgumentParser(
        description="Convert bounding boxes to integer-floor center coordinates."
    )


    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument(
        '--csv',
        '-c',
        type=str,
        help="Csv."
    )

    mode_group.add_argument(
        '--input',
        '-i',
        type=str,
        help="Input."
    )

    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help="Output."
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Verbose."
    )

    args = parser.parse_args()


    if args.input and not args.output:
        parser.error("--output is required when using --input")

    if args.output and not args.input:
        parser.error("--input is required when using --output")

    try:

        if args.input:
            print(f"Converting bounding boxes to centers...")
            print(f"  Input:  {args.input}")
            print(f"  Output: {args.output}")

            stats = calculate_centers_and_save(args.input, args.output, verbose=args.verbose)


            print("\nConversion complete")
            print(f"  Total entries:          {stats['num_entries']}")
            print(f"  Successfully converted: {stats['num_converted']}")

            if stats['num_skipped'] > 0:
                (f"{stats['num_skipped']}",)
                pass
                sys.exit(1)
            else:
                pass

            if args.verbose:
                print(f"\nOutput file created: {args.output}")
                output_size = Path(args.output).stat().st_size
                print(f"  File size: {output_size} bytes ({output_size / 1024:.1f} KB)")


        elif args.csv:
            print(f"Batch mode: Processing files from CSV")
            print(f"  CSV file: {args.csv}")

            batch_stats = process_csv_batch(args.csv, verbose=args.verbose)


            print("=" * 60)
            print("BATCH PROCESSING SUMMARY")
            print("=" * 60)
            print(f"Total file pairs:       {batch_stats['total_files']}")
            print(f"Successfully processed: {batch_stats['successful']}")
            print(f"Failed:                 {batch_stats['failed']}")
            print()
            print(f"Total neurons:          {batch_stats['total_entries']}")
            print(f"Successfully converted: {batch_stats['total_converted']}")
            (f"{batch_stats['total_skipped']}",)


            if batch_stats['failed_files']:
                (f"{len(batch_stats['failed_files'])}",)
                for input_path, error_msg in batch_stats['failed_files']:
                    print(f"  • {input_path}")
                    print(f"    Error: {error_msg}")
                sys.exit(1)
            else:
                print("\nAll files processed successfully")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
