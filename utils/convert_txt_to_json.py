import argparse
import json


def convert_to_numeric(value):
    """Try to convert a string value to a numeric one (int or float)."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_file_to_json(input_file):
    """Parse the content of the input file into a JSON format."""
    data_dict = {}
    with open(input_file, "r") as file:
        for line in file:
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                data_dict[key] = convert_to_numeric(value)
    return data_dict


def main():
    parser = argparse.ArgumentParser(description="Parse a text file into JSON.")
    parser.add_argument("input_file", type=str, help="Path to the input text file.")
    parser.add_argument(
        "-o", "--output", type=str, help="Path to the output JSON file."
    )

    args = parser.parse_args()

    # Parse the file
    parsed_data = parse_file_to_json(args.input_file)

    # Output the JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(parsed_data, f, indent=4)
        print(f"Saved to {args.output}")

    else:
        print("Not outpout directory.")


if __name__ == "__main__":
    main()
