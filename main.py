import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import AudioFileClip
from scipy.signal import welch

def generate_pds_graph(audio_file, output_svg):
    # Load the audio file
    audio = AudioFileClip(audio_file)

    # Extract the audio data
    audio_array = audio.to_soundarray(fps=44100)  # Adjust FPS as needed

    # Convert stereo to mono (if applicable)
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)

    # Calculate the power spectral density (PSD)
    f, Pxx = welch(audio_array, fs=audio.fps, nperseg=1024)

    # Plot the power density spectrum
    plt.figure()
    plt.plot(f, 10 * np.log10(Pxx))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title('Power Density Spectrum')
    plt.grid(True)

    # Save the plot as SVG
    plt.savefig(output_svg, format='svg')
    plt.close()

if __name__ == "__main__":
    # Example usage: python script_name.py input_audio.webm output_pds_graph.svg
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script_name.py input_audio.webm output_pds_graph.svg")
        sys.exit(1)

    input_audio = sys.argv[1]
    output_svg = sys.argv[2]

    generate_pds_graph(input_audio, output_svg)
    print("Power Density Spectrum graph saved as", output_svg)
