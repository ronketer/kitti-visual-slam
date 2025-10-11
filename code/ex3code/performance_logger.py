import time

class TimeLogger:
    def __init__(self):
        self.timings = {}
        self.start_times = {}
        self.operation_order = []
        
    def start_operation(self, name):
        self.start_times[name] = time.perf_counter()
        
    def end_operation(self, name):
        end_time = time.perf_counter()
        if name not in self.timings:
            self.timings[name] = []
            if name not in self.operation_order:
                self.operation_order.append(name)
        self.timings[name].append(end_time - self.start_times[name])
        
    def get_report(self):
        print("\nOperation Times (in chronological order):")
        iteration_times = []
        
        # First display non-iteration operations
        for operation in self.operation_order:
            times = self.timings[operation]
            if operation.startswith("Iteration "):
                iteration_times.extend(times)
                continue
                
            # Handle nested operations with indentation
            indent = "  " if "/" in operation else ""
            avg_time = sum(times) / len(times)
            print(f"{indent}- {operation}: {avg_time:.6f} seconds (called {len(times)} times)")
            
        if iteration_times:
            print("\nIteration Statistics:")
            avg_iter_time = sum(iteration_times) / len(iteration_times)
            print(f"- Average iteration time: {avg_iter_time:.6f} seconds")
            
            # Use only iteration times for total time calculation
            total_time = sum(iteration_times)
            print(f"Total Time: {total_time:.4f} seconds")
