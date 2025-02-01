import cProfile
import pstats
import psutil
from time import time, sleep
import os
import threading
import argparse
import audio_recorder
import matplotlib.pyplot as plt
import numpy as np

# Wrapper for profiling and CPU usage measurement
def profile_execution(stats_file):
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            process = psutil.Process()
            start_time = time()
            cpu_start = process.cpu_percent(interval=None)

            result = func(*args, **kwargs)

            cpu_end = process.cpu_percent(interval=None)
            elapsed_time = time() - start_time
            profiler.disable()

            # Ensure directory for stats file exists
            stats_dir = os.path.dirname(stats_file)
            if stats_dir and not os.path.exists(stats_dir):
                os.makedirs(stats_dir)

            # Save profiling statistics to file
            with open(stats_file, 'w') as f:
                stats = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
                stats.print_stats()
                f.write(f"\nExecution Time: {elapsed_time:.2f} seconds\n")
                f.write(f"CPU Usage: {cpu_end - cpu_start:.2f}%\n")

            print(f"Profiling data saved to {stats_file}")
            return result
        return wrapper
    return decorator

# Measure inference time and CPU usage
import model_w_profilling
inference_times = []

def classify_audio_with_profiling(interpreter, recorder, num_frames_hop):
    process = psutil.Process()
    start_time = time()
    cpu_start = process.cpu_percent(interval=None)

    model_w_profilling.classify_audio(recorder=recorder, interpreter=interpreter, num_frames_hop=num_frames_hop)

    cpu_end = process.cpu_percent(interval=None)
    end_time = time()

    inference_time = end_time - start_time
    cpu_usage = cpu_end - cpu_start

    inference_times.append((inference_time, cpu_usage))


# Modify main function in run_model.py
def main_with_profiling(stats_file, duration):
    stop_flag = threading.Event()

    @profile_execution(stats_file=stats_file)
    def wrapped_main():
        def stop_after_timeout():
            sleep(duration)
            print("Execution time exceeded, saving stats and stoping main...")
            # Calculate statistics
            times = [t[0] for t in inference_times]
            cpu_usages = [t[1] for t in inference_times]
            avg_time = sum(times) / len(times)
            avg_cpu = sum(cpu_usages) / len(cpu_usages)
            min_time = min(times)
            max_time = max(times)
            inferences = np.arange(len(inference_times))
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('Inferencia')
            ax1.set_ylabel('Tiempo de Inferencia (s)', color=color)
            ax1.plot(inferences, times, label='Tiempo de Inferencia', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('Porcentaje de Uso de la CPU (%)', color=color)  # we already handled the x-label with ax1
            ax2.plot(inferences, cpu_usages, label='Porcentaje de Uso de la CPU', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            fig.tight_layout()  # to make sure the right y-label is not slightly clipped
            plt.title('')
            plt.savefig("profiling_output/NO_CORAL_plot_5mins.png")
            plt.close()
            plt.savefig("profiling_output/CORAL")
            plt.close()
            # Save statistics to file
            stats_file = "profiling_output/NO_CORAL_detailed_stats_5mins.txt"
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            with open(stats_file, 'w') as f:
                f.write(f"Inference Statistics:\n")
                f.write(f"Average Inference Time: {avg_time:.4f}s\n")
                f.write(f"Minimum Inference Time: {min_time:.4f}s\n")
                f.write(f"Maximum Inference Time: {max_time:.4f}s\n")
                f.write(f"Average CPU Usage: {avg_cpu:.2f}%\n")

            print(f"Detailed statistics saved to {stats_file}")
            stop_flag.set()

        # Run the timeout in a separate thread
        timer_thread = threading.Thread(target=stop_after_timeout)
        timer_thread.start()

        try:
            # Modify the run_model.main loop to check stop_flag
            run_model_with_stop(stop_flag)
        except KeyboardInterrupt:
            print("Execution interrupted by user.")

    wrapped_main()

def run_model_with_stop(stop_flag):
    parser = argparse.ArgumentParser()
    model_w_profilling.add_model_flags(parser)
    args = parser.parse_args()
    interpreter = model_w_profilling.make_interpreter(args.model_file)
    interpreter.allocate_tensors()
    mic = args.mic if args.mic is None else int(args.mic)
    AUDIO_SAMPLE_RATE_HZ = 16000
    downsample_factor = 1
    if AUDIO_SAMPLE_RATE_HZ == 48000:
        downsample_factor = 3
    # Most microphones support this
    # Because the model expects 16KHz audio, we downsample 3 fold
    recorder = audio_recorder.AudioRecorder(
        AUDIO_SAMPLE_RATE_HZ,
        downsample_factor=downsample_factor,
        device_index=mic)

    while not stop_flag.is_set():
        try:
            classify_audio_with_profiling(interpreter=interpreter, recorder=recorder, num_frames_hop=int(args.num_frames_hop))
        except Exception as e:
            print(f"Error: {e}")
            break

def add_model_flags(parser):
  parser.add_argument(
      "--model_file",
      help="File path of TFlite model.",
      default="models/model_quant_2.tflite")
  parser.add_argument("--mic", default=None,
                      help="Optional: Input source microphone ID.")
  parser.add_argument(
      "--num_frames_hop",
      default=1,
      help="Optional: Number of frames to wait between model inference "
      "calls.")
  parser.add_argument(
      "--sample_rate_hz",
      default=16000,
      help="Optional: Sample Rate. The model expects 16000. "
      "However you may alternative sampling rate that may or may not work."
      "If you specify 48000 it will be downsampled to 16000.")
  
if __name__ == "__main__":
    print("Profiling execution with CORAL...")
    main_with_profiling(stats_file="profiling_output/NO_Coral_stats_5mins.txt", duration=300)

    # Now disable Coral by replacing interpreter
    """print("Profiling execution without CORAL...")
    def make_interpreter_no_coral(model_file):
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=model_file)

    model.make_interpreter = make_interpreter_no_coral
    main_with_profiling(stats_file="profiling_output/no_coral_stats.txt", duration=60)"""
