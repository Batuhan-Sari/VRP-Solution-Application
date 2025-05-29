import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import queue
import time

from file_parser import read_and_create_instance
from algorithms import SimulatedAnnealingSolver, TabuSearchSolver, GeneticAlgorithmSolver
from data_structures import ProblemInstance, Solution

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class VRPTW_GUI:
    def __init__(self, master):
        self.master = master
        master.title("VRPTW Solver")
        master.geometry("1200x800") 

        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1) 
        master.grid_columnconfigure(1, weight=3) 

        self.left_frame = ttk.Frame(master, padding="10")
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        for i in range(12): 
            self.left_frame.grid_rowconfigure(i, pad=7)
        self.left_frame.grid_rowconfigure(11, weight=1) 

        self.right_frame = ttk.Frame(master, padding="10")
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=2) 
        self.right_frame.grid_rowconfigure(1, weight=1) 
        self.right_frame.grid_columnconfigure(0, weight=1)

        # --- Graph Area ---
        self.graph_canvas_frame = ttk.LabelFrame(self.right_frame, text="Convergence Graph")
        self.graph_canvas_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ax.set_title("Convergence Graph")
        self.ax.set_xlabel("Time (s) or Iteration")
        self.ax.set_ylabel("Objective Value")
        self.ax.grid(True)
        self.canvas.draw() 

        # --- Output Text Area ---
        self.output_text_frame = ttk.LabelFrame(self.right_frame, text="Solution Output")
        self.output_text_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.output_text = tk.Text(self.output_text_frame, wrap=tk.WORD, height=12)
        self.output_text.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.output_text.insert(tk.END, "Solution details will appear here...")
        self.output_text.config(state=tk.DISABLED)

        # --- Instance and Algorithm Variables ---
        self.selected_instance_path = None
        self.selected_instance_file = tk.StringVar()
        self.selected_algorithm = tk.StringVar()
        self.problem_instance: ProblemInstance | None = None 
        self.solver_thread = None
        self.solver_queue = queue.Queue()

        # --- Global Parameters ---
        self.time_limit_seconds = tk.IntVar(value=60)
        self.ga_num_generations = tk.IntVar(value=100000)

        self._create_control_widgets()
        self.selected_algorithm.trace_add("write", self._on_algorithm_selected_update_solve_button) 

    def _create_control_widgets(self):
        """Creates widgets for the left control panel."""
        current_row = 0

        # --- Instance Selection ---
        ttk.Label(self.left_frame, text="Select Instance").grid(row=current_row, column=0, sticky="w", padx=5)
        current_row += 1
        
        self.instance_file_label = ttk.Label(self.left_frame, textvariable=self.selected_instance_file, wraplength=200)
        self.instance_file_label.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5)
        self.selected_instance_file.set("")
        current_row += 1

        self.select_instance_button = ttk.Button(self.left_frame, text="Select Instance", command=self._select_instance_file)
        self.select_instance_button.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=(0,10))
        current_row += 1

        # --- Algorithm Selection ---
        ttk.Label(self.left_frame, text="Select Solution Method").grid(row=current_row, column=0, sticky="w", padx=5)
        current_row += 1
        
        self.algorithm_combobox = ttk.Combobox(self.left_frame, textvariable=self.selected_algorithm, 
                                               values=["SimulatedAnnealing", "TabuSearch", "GeneticAlgorithm"],
                                               state=tk.DISABLED) 
        self.algorithm_combobox.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=(0,10))
        current_row += 1

        # --- Parameters Button ---
        self.set_params_button = ttk.Button(self.left_frame, text="Set Parameters", command=self._open_parameters_window,
                                            state=tk.DISABLED) 
        self.set_params_button.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=(0,10))
        current_row += 1
        
        # --- Solve Button ---
        self.solve_button = ttk.Button(self.left_frame, text="Solve", command=self._solve_problem, state=tk.DISABLED)
        self.solve_button.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=(20,10)) 
        current_row += 1

        # --- Status Label (at the bottom) ---
        self.status_label_var = tk.StringVar()
        self.status_label = ttk.Label(self.left_frame, textvariable=self.status_label_var, relief=tk.SUNKEN, anchor="w")
        self.status_label.grid(row=11, column=0, columnspan=2, sticky="sew", padx=5, pady=5) 
        self.status_label_var.set("Please select an instance file to begin.")

    def _on_algorithm_selected_update_solve_button(self, *args):
        if self.selected_instance_file.get() and self.selected_instance_file.get() != "No instance selected." and \
           self.selected_algorithm.get():
            self.solve_button.config(state=tk.NORMAL)
        else:
            self.solve_button.config(state=tk.DISABLED)

    def _select_instance_file(self):
        filepath = filedialog.askopenfilename(
            title="Select VRPTW Instance File",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
            initialdir="./"
        )
        if filepath:
            filename = os.path.basename(filepath)
            self.selected_instance_path = filepath
            self.selected_instance_file.set(f"You selected {filename}.")
            self.status_label_var.set(f"You selected {filename}.")
            self.algorithm_combobox.config(state="readonly")
            self.set_params_button.config(state=tk.NORMAL)
            self.problem_instance = None 
            print(f"Instance file selected: {filepath}")
        else:
            self.selected_instance_path = None
            self.selected_instance_file.set("")
            self.status_label_var.set("Instance selection cancelled. Please select an instance file.")
            self.algorithm_combobox.config(state=tk.DISABLED)
            self.algorithm_combobox.set("") 
            self.set_params_button.config(state=tk.DISABLED)
            self.solve_button.config(state=tk.DISABLED)
            self.problem_instance = None
        
        self._on_algorithm_selected_update_solve_button()

    def _open_parameters_window(self):
        if hasattr(self, 'param_window') and self.param_window.winfo_exists():
            self.param_window.destroy() 

        self.param_window = tk.Toplevel(self.master)
        self.param_window.title("Parameters")
        self.param_window.transient(self.master) 
        self.param_window.grab_set() 

        frame = ttk.Frame(self.param_window, padding="10")
        frame.pack(expand=True, fill=tk.BOTH)

        current_row = 0
        ttk.Label(frame, text="Time limit (sec):").grid(row=current_row, column=0, sticky="w", pady=2)
        time_limit_entry = ttk.Entry(frame, textvariable=self.time_limit_seconds, width=10)
        time_limit_entry.grid(row=current_row, column=1, sticky="ew", pady=2)
        current_row += 1

        def _validate_and_save_params():
            try:
                time_val = self.time_limit_seconds.get()
                if not (60 <= time_val <= 300):
                    messagebox.showwarning("Warning", "Time limit must be between 60 and 300 seconds. Defaulting to 60.", parent=self.param_window)
                    self.time_limit_seconds.set(60)
                return True
            except tk.TclError as e:
                messagebox.showerror("Error", f"Invalid parameter value. Please enter a valid number. Details: {e}", parent=self.param_window)
                self.time_limit_seconds.set(60)
                return False
            except Exception as e:
                messagebox.showerror("Error", f"An unexpected error occurred during parameter validation: {e}", parent=self.param_window)
                self.time_limit_seconds.set(60)
                return False

        def _save_params_and_close():
            if not _validate_and_save_params():
                return 
            self.status_label_var.set(f"Time limit set to {self.time_limit_seconds.get()} seconds.")
            self.param_window.destroy()

        ok_button = ttk.Button(frame, text="OK", command=_save_params_and_close)
        ok_button.grid(row=current_row, column=0, columnspan=2, pady=20)

        self.param_window.protocol("WM_DELETE_WINDOW", self.param_window.destroy)
        self.param_window.bind('<Escape>', lambda e: self.param_window.destroy()) 

    def _update_gui_before_solve(self):
        self.solve_button.config(state=tk.DISABLED)
        self.select_instance_button.config(state=tk.DISABLED)
        self.algorithm_combobox.config(state=tk.DISABLED)
        self.set_params_button.config(state=tk.DISABLED)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.ax.clear() 
        self.ax.set_title("Convergence Graph")
        self.ax.set_xlabel("Time (s) or Iteration")
        self.ax.set_ylabel("Objective Value")
        self.ax.grid(True)
        self.canvas.draw()
        self.master.update_idletasks()

    def _update_gui_after_solve(self):
        self.solve_button.config(state=tk.NORMAL)
        self.select_instance_button.config(state=tk.NORMAL)
        self.algorithm_combobox.config(state="readonly")
        self.set_params_button.config(state=tk.NORMAL)
        self.status_label_var.set("Ready.")

    def _solve_problem(self):
        instance_path = self.selected_instance_path
        algorithm_name = self.selected_algorithm.get()
        time_limit = self.time_limit_seconds.get()

        if not instance_path or not os.path.exists(instance_path):
            messagebox.showerror("Error", "Please select a valid problem instance file.")
            return
        if not algorithm_name:
            messagebox.showerror("Error", "Please select a solution algorithm.")
            return
        
        self._update_gui_before_solve()
        self.status_label_var.set(f"Loading instance and starting {algorithm_name}...")
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, f"Attempting to solve: {os.path.basename(instance_path)}\nAlgorithm: {algorithm_name}\nTime Limit: {time_limit} seconds\n\n")
        self.output_text.config(state=tk.DISABLED)
        self.master.update_idletasks()

        self.solver_thread = threading.Thread(
            target=self._run_solver_task,
            args=(instance_path, algorithm_name, time_limit),
            daemon=True 
        )
        self.solver_thread.start()
        self.master.after(100, self._check_solver_queue) 

    def _run_solver_task(self, instance_path: str, algorithm_name: str, time_limit: int):
        """The actual task performed by the solver thread."""
        try:
            self.problem_instance = read_and_create_instance(instance_path)
            if not self.problem_instance:
                raise ValueError("Failed to parse or create problem instance.")

            solver = None
            if algorithm_name == "SimulatedAnnealing":
                solver = SimulatedAnnealingSolver(
                    problem_instance=self.problem_instance,
                    time_limit_seconds=time_limit
                )
            elif algorithm_name == "TabuSearch":
                solver = TabuSearchSolver(
                    problem_instance=self.problem_instance,
                    time_limit_seconds=time_limit
                )
            elif algorithm_name == "GeneticAlgorithm":
                solver = GeneticAlgorithmSolver(
                    problem_instance=self.problem_instance,
                    time_limit_seconds=time_limit,
                    num_generations=100000
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")

            solution = solver.solve() 
            self.solver_queue.put({"solution": solution, "solver_instance": solver, "algorithm_name": algorithm_name, "error": None})
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in solver thread: {e}\n{error_details}")
            self.solver_queue.put({"solution": None, "solver_instance": None, "algorithm_name": algorithm_name, "error": str(e), "details": error_details})

    def _check_solver_queue(self):
        try:
            result = self.solver_queue.get_nowait()
            self._display_solution_output(result)
            if result.get("solver_instance") and not result.get("error"):
                self._display_convergence_graph(result["solver_instance"], result["algorithm_name"])
            
            self._update_gui_after_solve()
            if result.get("error"):
                self.status_label_var.set(f"Error during {result['algorithm_name']}. Check output.")
            else:
                self.status_label_var.set(f"{result['algorithm_name']} finished. Check output and graph.")

        except queue.Empty:
            if self.solver_thread and self.solver_thread.is_alive():
                self.status_label_var.set(f"Solving with {self.selected_algorithm.get()}... Please wait.")
                self.master.after(200, self._check_solver_queue) 
            else:
                self.status_label_var.set("Solver finished or encountered an issue.")
                self._update_gui_after_solve()
        except Exception as e: 
            import traceback
            error_info = traceback.format_exc()
            messagebox.showerror("GUI Error", f"An error occurred while processing solver results: {e}\n{error_info}")
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, f"\nGUI ERROR during result processing: {e}\n{error_info}\n")
            self.output_text.config(state=tk.DISABLED)
            self._update_gui_after_solve()
            self.status_label_var.set("Error processing results.")

    def _display_solution_output(self, result_dict: dict):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END) 

        instance_name = os.path.basename(self.selected_instance_file.get()) if self.selected_instance_file.get() else "N/A"
        algorithm_name = result_dict.get("algorithm_name", "N/A")

        self.output_text.insert(tk.END, f"--- Solver Run Report ---\n")
        self.output_text.insert(tk.END, f"Instance: {instance_name}\n")
        self.output_text.insert(tk.END, f"Algorithm: {algorithm_name}\n")
        self.output_text.insert(tk.END, f"Time Limit Set: {self.time_limit_seconds.get()}s\n\n")

        if result_dict.get("error"):
            self.output_text.insert(tk.END, f"ERROR during solving process:\n{result_dict['error']}\n")
            if result_dict.get("details"):
                 self.output_text.insert(tk.END, f"\nDetails:\n{result_dict['details']}\n")
            self.output_text.config(state=tk.DISABLED)
            return

        solution: Solution | None = result_dict.get("solution")
        solver_instance = result_dict.get("solver_instance") 

        if solver_instance and hasattr(solver_instance, 'start_time') and solver_instance.start_time > 0:
            time_elapsed = time.time() - solver_instance.start_time
            self.output_text.insert(tk.END, f"Actual Execution Time: {time_elapsed:.2f}s\n")
        else:
            self.output_text.insert(tk.END, f"Actual Execution Time: Not available\n")

        if solution:
            self.output_text.insert(tk.END, f"Solution Feasible: {solution.is_feasible}\n")
            self.output_text.insert(tk.END, f"Total Distance: {solution.total_distance:.2f}\n")
            self.output_text.insert(tk.END, f"Number of Routes: {len(solution.routes)}\n\n")
            
            if solution.route_details:
                for i, route_detail in enumerate(solution.route_details):
                    depot_id = self.problem_instance.depot.id if self.problem_instance else 0
                    route_cust_ids_display = [depot_id] + \
                                             route_detail.get('customer_ids', []) + \
                                             [depot_id]
                    self.output_text.insert(tk.END, f"  Route {i+1}: {' -> '.join(map(str, route_cust_ids_display))}\n")
                    self.output_text.insert(tk.END, f"    Distance: {route_detail.get('route_distance', 0.0):.2f}\n")
                    self.output_text.insert(tk.END, f"    Load: {route_detail.get('route_load', 0.0)}\n")
                    if route_detail.get('capacity_violation', False):
                        self.output_text.insert(tk.END, f"    CAPACITY VIOLATED\n")
                    if route_detail.get('time_violation', False):
                        self.output_text.insert(tk.END, f"    TIME VIOLATED\n")
                    self.output_text.insert(tk.END, "\n")
            else:
                self.output_text.insert(tk.END, "No routes in the solution or details not available.\n")
        else:
            self.output_text.insert(tk.END, "No solution object returned from solver.\n")
        
        self.output_text.insert(tk.END, "\n--- End of Report ---\n")
        self.output_text.config(state=tk.DISABLED)

    def _display_convergence_graph(self, solver_instance, algorithm_name: str):
        self.ax.clear()
        if solver_instance and hasattr(solver_instance, 'convergence_data') and solver_instance.convergence_data:
            valid_convergence_data = [
                (t, val) for t, val in solver_instance.convergence_data 
                if isinstance(val, (int, float)) and val != float('inf') and val != float('-inf') and not (isinstance(val, float) and val != val) 
            ]
            if not valid_convergence_data:
                print(f"No valid (finite) objective values in convergence data for {algorithm_name}.")
                self.ax.text(0.5, 0.5, 'No valid convergence data to plot.', horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes)
            else:
                times_or_iterations = [item[0] for item in valid_convergence_data]
                objective_values = [item[1] for item in valid_convergence_data]
                self.ax.plot(times_or_iterations, objective_values, marker='.', linestyle='-')
                self.ax.set_title(f"Convergence of {algorithm_name} for {self.problem_instance.name if self.problem_instance else ''}")
        else:
            self.ax.text(0.5, 0.5, 'No convergence data available.', horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes)
        
        self.ax.set_xlabel("Time (seconds) or Iteration")
        self.ax.set_ylabel("Total Distance (Objective Value)")
        self.ax.grid(True)
        self.fig.tight_layout() 
        self.canvas.draw()

def main():
    root = tk.Tk()
    gui = VRPTW_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 