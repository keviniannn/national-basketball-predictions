import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox

from data_generation import fetch_player_data
from over_under_model import generate_player_model
from predict_over_under import predict_over_under

def build_stat_expr():
    parts = []
    if pts_var.get(): parts.append("PTS")
    if reb_var.get(): parts.append("REB")
    if ast_var.get(): parts.append("AST")
    return "+".join(parts)

def run_prediction():
    player = player_entry.get().strip()
    try:
        line = float(line_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Line must be a number.")
        return

    stat_expr = build_stat_expr()
    if not stat_expr:
        messagebox.showerror("Input Error", "Select at least one stat (PTS, REB, AST).")
        return

    try:
        full_name = fetch_player_data(player, seasons=['2023-24', '2024-25'], force_refresh=refresh_var.get())
        generate_player_model(full_name, line=line, stat_expr=stat_expr, games=10, force_retrain=refresh_var.get())
        prob = predict_over_under(full_name, stat_expr=stat_expr, line=line, games=10)
        if prob is not None:
            prediction = "OVER" if prob > 0.5 else "UNDER"
            result_label.config(
                text=f"{prediction} ({prob:.2f})",
                bootstyle="success" if prediction == "OVER" else "danger"
            )
    except Exception as e:
        messagebox.showerror("Error", str(e))

app = ttk.Window(title="NBA Over/Under Predictor", themename="darkly", size=(400, 400))
app.resizable(False, False)

main = ttk.Frame(app, padding=15)
main.pack(fill="both", expand=True)

# player input
player_group = ttk.Labelframe(main, text="Player Name", padding=10)
player_group.pack(fill="x", pady=5)
player_entry = ttk.Entry(player_group)
player_entry.pack(fill="x")

# stat selection
stat_group = ttk.Labelframe(main, text="Stat Categories", padding=10)
stat_group.pack(fill="x", pady=5)

pts_var = ttk.BooleanVar(value=True)
reb_var = ttk.BooleanVar(value=True)
ast_var = ttk.BooleanVar(value=True)

ttk.Checkbutton(stat_group, text="PTS", variable=pts_var).pack(side="left", padx=10)
ttk.Checkbutton(stat_group, text="REB", variable=reb_var).pack(side="left", padx=10)
ttk.Checkbutton(stat_group, text="AST", variable=ast_var).pack(side="left", padx=10)

# line input
line_group = ttk.Labelframe(main, text="Over/Under Line", padding=10)
line_group.pack(fill="x", pady=5)
line_entry = ttk.Entry(line_group)
line_entry.pack(fill="x")

# refresh
refresh_var = ttk.BooleanVar()
ttk.Checkbutton(main, text="Force Refresh Data/Model", variable=refresh_var).pack(anchor="w", pady=(5, 10))

# run button
ttk.Button(main, text="Run Prediction", command=run_prediction, bootstyle="primary").pack(fill="x")

# result label
result_label = ttk.Label(main, text="", font=("Segoe UI", 12, "bold"), anchor="center", justify="center")
result_label.pack(pady=15)

app.mainloop()