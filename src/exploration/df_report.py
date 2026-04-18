import pandas as pd
import sweetviz as sv

df = pd.read_csv("./sample_stream_output.csv")
report = sv.analyze(df)

# Save to HTML
report.show_html("report.html")