import glob
for file in glob.glob("/workspace/wiki/learn-cleanrl/*tutorial.py"):
    with open(file, "r") as f:
        text = f.read()
    
    bad = """        if "v_loss" in locals() and "pg_loss" in locals():
            tracker.log_metrics("losses", {"""
    good = """        if "v_loss" in locals() and "pg_loss" in locals():
            tracker.log_metrics("losses", {"""
            
    text = text.replace(
"""        if "v_loss" in locals() and "pg_loss" in locals():
            tracker.log_metrics("losses", {""",
"""        if "v_loss" in locals() and "pg_loss" in locals():
            tracker.log_metrics("losses", {"""
    ) # oops

    import re
    text = re.sub(
        r'        if "v_loss" in locals\(\) and "pg_loss" in locals\(\):\n            tracker\.log_metrics\("losses", \{',
        r'        if "v_loss" in locals() and "pg_loss" in locals():\n            tracker.log_metrics("losses", {',
        text
    )
    
    # Wait, the error is: tracker.log_metrics... is on line 314
    # What was inserted?
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if 'if "v_loss" in locals() and "pg_loss" in locals():' in line:
            # Let's fix the block indentation
            pass

##############################
