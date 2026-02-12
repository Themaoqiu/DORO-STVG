import json
import re
from typing import Dict, Any, Optional

def parse_time_to_seconds(time_str: str) -> tuple:
    """
    Parse time string to start and end seconds.
    
    Args:
        time_str: Time string like "0:01-0:04" or "01:47-01:50" or "18:33-18:48"
    
    Returns:
        tuple: (start_seconds, end_seconds)
    """
    start_str, end_str = time_str.split('-')
    
    def time_to_seconds(t_str: str) -> int:
        parts = t_str.split(':')
        if len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid time format: {t_str}")
    
    return time_to_seconds(start_str), time_to_seconds(end_str)


class JSONParser:
    """Utility class for parsing and extracting JSON from text responses."""

    @staticmethod
    def parse(response: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from a response string.

        Args:
            response: String potentially containing JSON

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        # Try to match JSON in a code block first
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code block, try the entire string
            json_str = response
        try:
            parsed_data = json.loads(json_str)
            if isinstance(parsed_data, dict):
                return parsed_data
            return None
        except json.JSONDecodeError:
            return None