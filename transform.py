import os
from shutil import copyfile
from scikits.audiolab import Sndfile, Format


def transform_wav(wav_file, output):
    filename = os.path.basename(wav_file)
    pathname = os.path.dirname(wav_file)
    speaker = os.path.basename(pathname)
    dr = os.path.basename(os.path.dirname(pathname))
    output_dir = os.path.join(output, dr, speaker)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, filename)

    f = Sndfile(wav_file, 'r')
    data = f.read_frames(f.nframes)
    fs = f.samplerate
    nc = f.channels
    enc = f.encoding
    
    wav_format = Format('wav')
    f_out = Sndfile(output_file, 'w', wav_format, nc, fs)
    f_out.write_frames(data)
    f.close()
    f_out.close()


def copy_phn(phn_file, output):
    filename = os.path.basename(phn_file)
    pathname = os.path.dirname(phn_file)
    speaker = os.path.basename(pathname)
    dr = os.path.basename(os.path.dirname(pathname))
    output_file = os.path.join(output, dr, speaker, filename)

    copyfile(phn_file, output_file)


