import os
import fnmatch
import argparse
from collections import defaultdict


def find_in_yml_file(old_str, filepath):
    """
    Find all occurrences of old_str in a single.yml file and return their locations.

    Args:
        old_str (str): String to find
        filepath (str): Path to the.yml file

    Returns:
        dict: Dictionary with line numbers as keys and list of (start_pos, end_pos) as values
    """
    occurrences = defaultdict(list)

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                start = 0
                while True:
                    pos = line.find(old_str, start)
                    if pos == -1:
                        break
                    occurrences[line_num].append((pos, pos + len(old_str)))
                    start = pos + 1

    except Exception as e:
        print(f"Error reading {filepath}: {str(e)}")

    return occurrences


def display_single_change(filepath, line_num, line, positions, old_str, new_str):
    """
    Display a single change that would be made and prompt for confirmation.
    """
    print(f"\nFile: {filepath}")
    print(f"Line {line_num}:")
    print(line.rstrip())

    # Show markers for each occurrence
    marker_line = [' '] * len(line)
    for start, end in positions:
        for i in range(start, end):
            if i < len(marker_line):
                marker_line[i] = '^'

    print(''.join(marker_line))
    print(f"Change '{old_str}' to '{new_str}'? ([y]/n/nf(skip file)/q): ")
    return input().strip().lower()


def preview_replacements(old_str, new_str, directory='.', single_file=None):
    """
    Preview all replacements that would be made.

    Args:
        old_str (str): String to be replaced
        new_str (str): String to replace with
        directory (str): Root directory to search from
        single_file (str): If specified, only process this file

    Returns:
        tuple: (total_replacements, list_of_files) where list_of_files contains
               (filepath, occurrences_dict, file_content) tuples
    """
    total_replacements = 0
    files_to_process = []

    if single_file:
        # Process just the single file
        if os.path.isfile(single_file) and single_file.endswith('.yml'):
            try:
                with open(single_file, 'r', encoding='utf-8') as file:
                    content = file.readlines()
                occurrences = find_in_yml_file(old_str, single_file)
                if occurrences:
                    count = sum(len(positions) for positions in occurrences.values())
                    total_replacements += count
                    files_to_process.append((single_file, occurrences, content))
            except Exception as e:
                print(f"Error reading {single_file}: {str(e)}")
    else:
        # Walk through all directories and subdirectories
        for root, dirs, files in os.walk(directory):
            for filename in fnmatch.filter(files, '*.yml'):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.readlines()
                    occurrences = find_in_yml_file(old_str, filepath)
                    if occurrences:
                        count = sum(len(positions) for positions in occurrences.values())
                        total_replacements += count
                        files_to_process.append((filepath, occurrences, content))
                except Exception as e:
                    print(f"Error reading {filepath}: {str(e)}")

    return total_replacements, files_to_process


def display_preview(total_replacements, files_to_process, old_str, new_str):
    """
    Display the preview of changes that would be made.
    """
    print(f"\nPreview of changes ({total_replacements} total replacements):")
    print("=" * 60)

    for filepath, occurrences, content in files_to_process:
        print(f"\nFile: {filepath}")

        for line_num, positions in sorted(occurrences.items()):
            line = content[line_num - 1].rstrip()
            print(f"  Line {line_num}:")
            print(f"  >{line}")

            # Show markers for each occurrence
            marker_line = [' '] * len(line)
            for start, end in positions:
                for i in range(start, end):
                    if i < len(marker_line):
                        marker_line[i] = '^'

            print(f"   {''.join(marker_line)}")

    print("\n" + "=" * 60)


def perform_replacements(old_str, new_str, files_to_process, interactive=False):
    """
    Actually perform the replacements that were previewed.

    Args:
        old_str (str): String to replace
        new_str (str): Replacement string
        files_to_process: List of files with occurrences
        interactive (bool): Whether to prompt for each change

    Returns:
        int: Total number of replacements made
    """
    total_replacements = 0

    for filepath, occurrences, content in files_to_process:
        try:
            if interactive:
                # In interactive mode, we need to process each change individually
                new_lines = content.copy()
                skip_file = False

                for line_num, positions in sorted(occurrences.items(), reverse=True):
                    if skip_file:
                        break

                    line = content[line_num - 1]
                    response = display_single_change(
                        filepath, line_num, line, positions, old_str, new_str
                    )

                    if response == 'y' or response == '':
                        # Replace all occurrences in this line
                        new_line = line.replace(old_str, new_str)
                        new_lines[line_num - 1] = new_line
                        total_replacements += len(positions)
                    elif response == 'n':
                        # Skip this change
                        continue
                    elif response == 'nf':
                        # Skip entire file
                        skip_file = True
                        continue
                    elif response == 'q':
                        # Quit entirely
                        return total_replacements

                if not skip_file:
                    # Write the modified content back to the file
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.writelines(new_lines)
                    print(f"Saved changes to {filepath}")

            else:
                # Non-interactive mode - replace all at once
                with open(filepath, 'r', encoding='utf-8') as file:
                    file_content = file.read()

                new_content = file_content.replace(old_str, new_str)

                if new_content != file_content:
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(new_content)

                    count = file_content.count(old_str)
                    total_replacements += count
                    print(f"Replaced {count} occurrences in {filepath}")

        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")

    return total_replacements


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Replace strings in .yml files')
    parser.add_argument('--old', '-o', required=True, help='String to be replaced')
    parser.add_argument('--new', '-n', required=True, help='String to replace with')
    parser.add_argument('--directory', '--dir', '-d', default='.',
                        help='Root directory to search (default: current directory)')
    parser.add_argument('--file', '-f', default=None,
                        help='Single file to process (default: None)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Prompt for each individual change')

    # Parse arguments
    args = parser.parse_args()

    # First, find all potential replacements
    total_replacements, files_to_process = preview_replacements(
        args.old, args.new, args.directory, args.file
    )

    if total_replacements == 0:
        print("\nNo occurrences found to replace.")
        exit()

    if args.interactive:
        # Interactive mode - process changes one by one
        print(f"\nFound {total_replacements} potential replacements in {len(files_to_process)} file(s)")
        print("Starting interactive replacement...")
        actual_replacements = perform_replacements(
            args.old, args.new, files_to_process, interactive=True
        )
        print(f"\nDone! {actual_replacements} replacements made.")
    else:
        # Non-interactive mode - show preview first
        display_preview(total_replacements, files_to_process, args.old, args.new)

        # Ask for confirmation
        print(f"\nThis will replace {total_replacements} occurrences of '{args.old}' with '{args.new}' "
              f"in {len(files_to_process)} file(s). Continue? (y/n): ")
        confirm = input().strip().lower()

        if confirm == 'y':
            actual_replacements = perform_replacements(
                args.old, args.new, files_to_process, interactive=False
            )
            print(f"\nDone! {actual_replacements} replacements made.")
        else:
            print("Operation cancelled.")