import numpy as np

def segmented_with_half_lenght_overlap(sequence, seg_size):
    '''Converts a list into a list of segments which overlap (with half length overlap size).'''
    assert(seg_size % 2 == 0)

    overlap_size = seg_size // 2
    return [sequence[i : i + seg_size] for i in range(0, len(sequence) - seg_size + 1, overlap_size)]

def trianglify(segment, up=True, down=True, mult_first_by_zero=False):
    '''Multiplies the given segment by a triangular pulse.'''
    assert(len(segment) % 2 == 0)

    half_seg_size = len(segment) // 2
    dy = 0 if mult_first_by_zero else 1 / half_seg_size

    ascent = np.linspace(dy, 1-dy, half_seg_size)
    descent = np.ones(ascent.shape) - ascent
    segment_t = np.array(segment, dtype=float)

    # 1st half, x in [0, w/2): f(x) = x / (w/2)
    if up: segment_t[:half_seg_size] *= ascent
    # 2nd half, x in [w/2, w]: f(x) = 2 - x / (w/2)
    if down: segment_t[half_seg_size:] *= descent

    return segment_t

def modulate_and_add(segmented, seg_size, original_len):
    '''Expects a list of overlapping segments (with half length overlap size) and 
       modulates and adds them to reconstruct the original sequence.'''
    assert(len(segmented[0]) == seg_size) # assert(all(len(segmented[i]) == seg_size for i in range(0, len(segmented))))

    # modulating 'segmented' by a triangular pulse
    modulated = [trianglify(segment, up=i > 0, down=i < len(segmented) - 1) 
                 for i, segment in enumerate(segmented)]
    
    # adding the modulated overlapping segments to reobtain the original sequence
    added = np.zeros(original_len)
    for i in range(0, len(modulated)):
        added[i * overlap_size : i * overlap_size + seg_size] += modulated[i]
    
    return added

###############################################################

def __equal_seq(seq1, seq2, max_abs_error=0.0001):
    return all(abs(seq1[i] - seq2[i]) < max_abs_error 
        for i in range(0, min(len(seq1), len(seq2))))

def __print_arr(arr, arr_name="arr", only_shape=False):
    try:
        print(f"{arr_name}.shape={arr.shape}")
        if not only_shape: print(f"{arr_name}={arr}\n")
    except:
        np_arr = np.array(arr)
        print(f"*{arr_name}.shape={np_arr.shape}")
        if not only_shape: print(f"*{arr_name}={np_arr}\n")

###############################################################

if __name__ == "__main__":
    import sys
    from scipy.io import wavfile

    src_fname = 'link.wav' if len(sys.argv) < 2 else sys.argv[1]
    dst_fname = 'link_procd.wav' if len(sys.argv) < 3 else sys.argv[2]
    sr, wave = wavfile.read(src_fname)
    print(f"src_fname={src_fname}\ndst_fname={dst_fname}\n")
    print(f"sr={sr}, wave.shape={wave.shape}, wave.dtype={wave.dtype}\n")

    # [...]
    original = np.copy(wave)
    seg_size = 32
    overlap_size = seg_size // 2

    # channel 1
    ch_1 = original[:, 0]
    segmented_ch_1 = segmented_with_half_lenght_overlap(ch_1, seg_size) # [...] -> [[..], ...]
    procd_ch_1 = modulate_and_add(segmented_ch_1, seg_size, original_len=ch_1.shape)
    print(f"equal channel 1? {__equal_seq(ch_1, original[:, 0])}")

    # channel 2
    ch_2 = original[:, 1]
    segmented_ch_2 = segmented_with_half_lenght_overlap(ch_2, seg_size) # [...] -> [[..], ...]
    procd_ch_2 = modulate_and_add(segmented_ch_2, seg_size, original_len=ch_2.shape)
    print(f"equal channel 2? {__equal_seq(ch_2, original[:, 1])}")

    # wave reconstruction
    procd_wave = np.array([(x, y) for x, y in zip(ch_1, ch_2)], dtype=np.int16)
    print(f"\nprocd_wave.shape={procd_wave.shape}, procd_wave.dtype={procd_wave.dtype}")

    print(f"\nsaving processed wave to {dst_fname}")
    wavfile.write(dst_fname, rate=sr, data=procd_wave)
