import logging
import re
import math
from typing import Union, Optional
from llama_index.core.tools import FunctionTool

logger = logging.getLogger(__name__)


class CalculatorTool:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.safe_functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'sqrt': math.sqrt,
            'pow': pow,
            'log': math.log,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e,
        }

    def calculate(self, expression: str) -> str:
        try:
            if not expression or not expression.strip():
                return "Error: Empty expression provided. Please provide a mathematical expression."

            cleaned_expr = self._clean_expression(expression)
            if not cleaned_expr:
                return "Error: Invalid mathematical expression."

            self.logger.info(f"Calculating: {cleaned_expr}")

            safe_dict = {
                "__builtins__": {},
                **self.safe_functions
            }

            result = eval(cleaned_expr, safe_dict)

            formatted_result = self._format_result(result, expression)

            self.logger.info(f"Calculation result: {cleaned_expr} = {result}")
            return formatted_result

        except ZeroDivisionError:
            return "Error: Division by zero is not allowed."
        except ValueError as e:
            return f"Error: Invalid value in calculation - {str(e)}"
        except SyntaxError:
            return "Error: Invalid mathematical expression syntax."
        except Exception as e:
            error_msg = f"Error in calculation: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def _clean_expression(self, expression: str) -> Optional[str]:
        expr = expression.strip()

        dangerous_patterns = [
            r'import\s+',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'__',
            r'getattr',
            r'setattr',
            r'delattr',
            r'globals',
            r'locals',
            r'vars',
            r'dir\s*\(',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, expr, re.IGNORECASE):
                self.logger.warning(f"Dangerous pattern detected in expression: {expr}")
                return None

        allowed_chars = re.match(r'^[0-9+\-*/().\s,a-zA-Z_]+$', expr)
        if not allowed_chars:
            return None

        return expr

    def _format_result(self, result: Union[int, float], original_expr: str) -> str:
        is_percentage = any(keyword in original_expr.lower()
                            for keyword in ['percent', '%', 'percentage'])

        if isinstance(result, float):
            if result.is_integer():
                formatted = str(int(result))
            elif abs(result) < 0.001:
                formatted = f"{result:.6f}"
            elif abs(result) < 1:
                formatted = f"{result:.4f}"
            else:
                formatted = f"{result:.2f}"
        else:
            formatted = str(result)

        if is_percentage and not formatted.endswith('%'):
            formatted += "%"

        return f"   Calculation Result: {formatted}"

    def calculate_percentage(self, part: float, total: float) -> str:
        try:
            if total == 0:
                return "Error: Cannot calculate percentage with total of zero."

            percentage = (part / total) * 100
            return f" Percentage: {percentage:.2f}%"

        except Exception as e:
            return f"Error calculating percentage: {str(e)}"

    def calculate_average(self, numbers: str) -> str:
        try:
            num_list = [float(x.strip()) for x in numbers.split(',') if x.strip()]

            if not num_list:
                return "Error: No valid numbers provided."

            average = sum(num_list) / len(num_list)
            return f"Average of {len(num_list)} numbers: {average:.2f}"

        except ValueError:
            return "Error: Invalid number format. Use comma-separated numbers like '1,2,3,4,5'"
        except Exception as e:
            return f"Error calculating average: {str(e)}"

    def calculate_statistics(self, numbers: str) -> str:
        try:
            num_list = [float(x.strip()) for x in numbers.split(',') if x.strip()]

            if not num_list:
                return "Error: No valid numbers provided."

            stats = {
                'count': len(num_list),
                'min': min(num_list),
                'max': max(num_list),
                'average': sum(num_list) / len(num_list),
                'sum': sum(num_list)
            }

            result_parts = [
                "   Statistical Summary:",
                f"   Count: {stats['count']}",
                f"   Minimum: {stats['min']}",
                f"   Maximum: {stats['max']}",
                f"   Average: {stats['average']:.2f}",
                f"   Sum: {stats['sum']:.2f}"
            ]

            return "\n".join(result_parts)

        except ValueError:
            return "Error: Invalid number format. Use comma-separated numbers like '1,2,3,4,5'"
        except Exception as e:
            return f"Error calculating statistics: {str(e)}"


def create_calculator_tool() -> FunctionTool:
    tool_instance = CalculatorTool()

    calculator_tool = FunctionTool.from_defaults(
        fn=tool_instance.calculate,
        name="calculate",
        description=(
            "Perform mathematical calculations and arithmetic operations. "
            "Use this tool for any numeric calculations like '10 + 5', 'sqrt(16)', "
            "percentage calculations '(20/100)*50', or any mathematical expressions. "
            "Can handle basic arithmetic, percentages, square roots, and other math functions."
        )
    )

    return calculator_tool


def create_percentage_tool() -> FunctionTool:
    tool_instance = CalculatorTool()

    percentage_tool = FunctionTool.from_defaults(
        fn=tool_instance.calculate_percentage,
        name="calculate_percentage",
        description=(
            "Calculate the percentage of a part relative to a total. "
            "Use this when you need to find what percentage one number represents of another."
        )
    )

    return percentage_tool


def create_average_tool() -> FunctionTool:
    tool_instance = CalculatorTool()

    average_tool = FunctionTool.from_defaults(
        fn=tool_instance.calculate_average,
        name="calculate_average",
        description=(
            "Calculate the average of a list of numbers. "
            "Provide numbers as comma-separated string like '1,2,3,4,5'."
        )
    )

    return average_tool


def create_statistics_tool() -> FunctionTool:
    tool_instance = CalculatorTool()

    stats_tool = FunctionTool.from_defaults(
        fn=tool_instance.calculate_statistics,
        name="calculate_statistics",
        description=(
            "Calculate basic statistics (count, min, max, average, sum) for a list of numbers. "
            "Useful for analyzing numeric data like years of experience, salaries, etc."
        )
    )

    return stats_tool