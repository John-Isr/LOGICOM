import os
import argparse

# Path to the API keys file relative to this script's location (utils/)
# Assumes API_keys is in the project root directory (parent of utils/)
API_KEYS_FILENAME = "API_keys"
API_KEYS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), API_KEYS_FILENAME)

EXPECTED_KEYS = {
    "OpenAI_API_key:": "OPENAI_API_KEY",
    "Google_API_key:": "GOOGLE_API_KEY"
}

def set_environment_variables_from_file(file_path: str):
    """Reads API keys from a file and sets them as environment variables."""
    
    keys_found_in_file = {}
    keys_set = []
    keys_already_set = []

    print(f"Attempting to read API keys from: {file_path}")

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): # Skip empty lines and comments
                    continue
                
                for prefix, env_var_name in EXPECTED_KEYS.items():
                    if line.startswith(prefix):
                        key_value = line[len(prefix):].strip()
                        if key_value:
                            keys_found_in_file[env_var_name] = key_value
                        else:
                            print(f"Warning: Found prefix '{prefix}' but key value is empty in {file_path}")
                        break # Move to next line once prefix is matched
    except FileNotFoundError:
        print(f"Error: API keys file not found at {file_path}")
        print("Please create the file or ensure it's in the correct location ('Reworked/' directory). You can use API_keys.template as a reference.")
        return # Exit if file not found
    except Exception as e:
        print(f"Error reading API keys file {file_path}: {e}")
        return # Exit on other read errors

    # Now set the environment variables based on found keys
    for env_var_name, key_value in keys_found_in_file.items():
        if os.getenv(env_var_name):
            print(f"{env_var_name} environment variable already set. Overwriting using value from {API_KEYS_FILENAME}.")
            keys_already_set.append(env_var_name)
        else:
            print(f"Setting {env_var_name} environment variable from {API_KEYS_FILENAME}.")
        os.environ[env_var_name] = key_value
        keys_set.append(env_var_name)

    # Check for expected keys not found in file
    for env_var_name in EXPECTED_KEYS.values():
        if env_var_name not in keys_found_in_file:
             if os.getenv(env_var_name):
                  print(f"{env_var_name} not found in {API_KEYS_FILENAME}, but environment variable is already set.")
                  keys_already_set.append(env_var_name)
             else:
                  print(f"Warning: {env_var_name} not found in {API_KEYS_FILENAME} and environment variable is not set.")

    print("\nSummary:")
    distinct_keys_set = list(set(keys_set))
    distinct_keys_already_present = list(set(keys_already_set) - set(keys_set)) # Ones already set but not overwritten now
    
    if distinct_keys_set:
        print(f"- Set/Overwrote environment variables: {', '.join(distinct_keys_set)}")
    if distinct_keys_already_present:
         print(f"- Environment variables already set (and not overwritten): {', '.join(distinct_keys_already_present)}")
    if not distinct_keys_set and not distinct_keys_already_present:
         print("- No API keys were successfully set from the file or found in the environment.")

def main():

    set_environment_variables_from_file(API_KEYS_PATH)

if __name__ == "__main__":
    main() 