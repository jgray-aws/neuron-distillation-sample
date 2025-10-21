from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from IPython.display import HTML
from typing import Union, Tuple

def prettyprint_python(path: str, 
                      line_numbers: Union[bool, str] = True,
                      line_range: Tuple[int, int] = None) -> HTML:
    """
    Reads a Python file and returns a formatted HTML display with syntax highlighting and optional line numbers.
    
    Args:
        path (str): Path to the Python file to be formatted
        line_numbers (Union[bool, str]): Controls line number display:
            - True: Shows line numbers (default)
            - False: Hides line numbers
            - 'inline': Shows inline line numbers
            - 'table': Shows line numbers in separate column
        line_range (Tuple[int, int], optional): Start and end line numbers to display (e.g., (10, 20))
    
    Returns:
        IPython.display.HTML: Formatted code with syntax highlighting and line numbers
    """
    try:
        # Read the file
        with open(path, 'r') as f:
            code = f.read()
        
        # Handle line range if specified
        if line_range:
            start, end = line_range
            code_lines = code.splitlines()
            if 1 <= start <= len(code_lines) and start <= end:
                code = '\n'.join(code_lines[start-1:end])
            else:
                raise ValueError(f"Invalid line range: {line_range}")
        
        # Configure formatter based on line number preference
        formatter_kwargs = {
            'style': 'monokai',  # Changed to monokai dark theme
            'linenos': line_numbers if isinstance(line_numbers, bool) else True,
            'lineanchors': 'line',
            'anchorlinenos': True,
        }
        
        # Add specific formatting for inline/table display
        if isinstance(line_numbers, str):
            if line_numbers.lower() == 'inline':
                formatter_kwargs['linenos'] = 'inline'
            elif line_numbers.lower() == 'table':
                formatter_kwargs['linenos'] = 'table'
        
        formatter = HtmlFormatter(**formatter_kwargs)
        
        # Generate HTML with custom CSS
        html_code = f"""
        <style>
        {formatter.get_style_defs('.highlight')}
        .highlight {{ 
            background: #272822; 
            padding: 10px; 
            text-align: left;
            border-radius: 5px;
        }}
        .highlight pre {{ 
            text-align: left;
            margin: 0;
            color: #f8f8f2;
        }}
        .linenos {{ 
            color: #75715e; 
            padding-right: 10px;
            text-align: right;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            user-select: none;
        }}
        .highlight .hll {{ background-color: #49483e }}
        .highlight .c {{ color: #75715e }} /* Comment */
        .highlight .err {{ color: #960050; background-color: #1e0010 }} /* Error */
        .highlight .k {{ color: #66d9ef }} /* Keyword */
        .highlight .l {{ color: #ae81ff }} /* Literal */
        .highlight .n {{ color: #f8f8f2 }} /* Name */
        .highlight .o {{ color: #f92672 }} /* Operator */
        .highlight .p {{ color: #f8f8f2 }} /* Punctuation */
        .highlight .s {{ color: #e6db74 }} /* String */
        .highlight .na {{ color: #a6e22e }} /* Name.Attribute */
        .highlight .nb {{ color: #f8f8f2 }} /* Name.Builtin */
        .highlight .nc {{ color: #a6e22e }} /* Name.Class */
        .highlight .no {{ color: #66d9ef }} /* Name.Constant */
        .highlight .nd {{ color: #a6e22e }} /* Name.Decorator */
        .highlight .ni {{ color: #f8f8f2 }} /* Name.Entity */
        .highlight .ne {{ color: #a6e22e }} /* Name.Exception */
        .highlight .nf {{ color: #a6e22e }} /* Name.Function */
        .highlight .nl {{ color: #f8f8f2 }} /* Name.Label */
        .highlight .nn {{ color: #f8f8f2 }} /* Name.Namespace */
        .highlight .nx {{ color: #a6e22e }} /* Name.Other */
        .highlight .py {{ color: #f8f8f2 }} /* Name.Property */
        .highlight .nt {{ color: #f92672 }} /* Name.Tag */
        .highlight .nv {{ color: #f8f8f2 }} /* Name.Variable */
        .highlight .w {{ color: #f8f8f2 }} /* Text.Whitespace */
        </style>
        {highlight(code, PythonLexer(), formatter)}
        """
        
        return HTML(html_code)
    
    except FileNotFoundError:
        print(f"Error: File '{path}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")
