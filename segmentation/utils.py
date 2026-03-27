from rich.console import Console
from rich.table import Table

def rich_table_to_latex(table: Table) -> str:
    """
    Converts a rich.table.Table object into a LaTeX tabular string.
    """
    # 1. Extract headers from the columns
    headers = [col.header for col in table.columns]
    
    # 2. Extract row data 
    # (Rich stores cell data in a protected _cells list inside each column)
    num_rows = len(table.columns[0]._cells) if table.columns else 0
    rows = []
    for i in range(num_rows):
        # Convert each cell to a string to strip out rich renderables if necessary
        row = [str(col._cells[i]) for col in table.columns]
        rows.append(row)
        
    # 3. Construct the LaTeX string
    num_cols = len(table.columns)
    column_alignment = "c" * num_cols # Defaulting to center alignment
    
    latex_lines = [
        f"\\begin{{tabular}}{{{column_alignment}}}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline"
    ]
    
    for row in rows:
        latex_lines.append(" & ".join(row) + " \\\\")
        
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}"
    ])
    
    return "\n".join(latex_lines)