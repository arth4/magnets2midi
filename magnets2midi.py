from flatspin.grid import Grid
from flatspin.data import load_output, read_table
import pretty_midi
import numpy as np


def addNotes(notes, scale, instrument, tempo=0.5):
    for i, note in enumerate(notes):
        if note >= len(scale):
            continue
        note_name = scale[note % len(scale)]
        note_number = pretty_midi.note_name_to_number(note_name)
        instrument.notes.append(pretty_midi.Note(
            velocity=100, pitch=note_number, start=0 + tempo*i, end=tempo + tempo*i))


def make_c_scale(num=5):
    return [letter+str(num) for letter in ['C', 'D', 'E', 'F', 'G', 'A', 'B']] + [f"C{num+1}"]


def dataset2song(ds, tempo=0.4):
    instruments_notes = dataset2notes(ds)
    intruments = np.random.choice(
        pretty_midi.constants.INSTRUMENT_MAP, len(instruments_notes), replace=False)
    octaves = np.random.choice(
        np.arange(2, 5), len(instruments_notes), replace=True)
    midi_object = pretty_midi.PrettyMIDI()
    for instr_name, scale_num, instr_notes in zip(intruments, octaves, instruments_notes):
        instr_program = pretty_midi.instrument_name_to_program(instr_name)
        instr = pretty_midi.Instrument(program=instr_program)
        c_scale = make_c_scale(scale_num)
        addNotes(instr_notes, c_scale, instr, tempo=tempo)
        midi_object.instruments.append(instr)
    return midi_object


def dataset2notes(ds, grid_size=(4, 4), scale_size=8):
    geom = read_table(ds.tablefile("geom"))
    poss = np.array(list(zip(geom["posx"].values, geom["posy"].values)))
    grid = Grid.fixed_grid(np.array(poss), grid_size)
    mag = load_output(ds, "mag", grid_size=grid_size, flatten=False)

    angle = vector_colors(mag[..., 0], mag[..., 1])  # [0,2pi)
    norm_angle = angle/(2*np.pi)  # [0,1)
    norm_angle *= scale_size  # [0,scale_size)
    norm_angle = norm_angle.round().astype(int)
    norm_angle[norm_angle >= scale_size] = 0
    intr_notes = [norm_angle[:, i, j]
                  for i in range(grid_size[0]) for j in range(grid_size[1])]
    return [filter_repeat_notes(i_n, scale_size) for i_n in intr_notes]


def filter_repeat_notes(notes, rest_note_num=8):
    """replace consecutive duplicates with a special rest note value"""
    notes = notes.copy()
    edif = np.concatenate(([1], np.ediff1d(notes)))
    notes[edif == 0] = rest_note_num
    return notes


def vector_colors(U, V):
    C = np.arctan2(V, U)  # color
    C[C < 0] = 2*np.pi + C[C < 0]
    return C


if __name__ == '__main__':
    from flatspin.cmdline import main_dataset_argparser, main_dataset

    parser = main_dataset_argparser("magnets2midi", True)
    parser.add_argument('--bpm', type=float, default=200,
                        help='beats per minute (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to use (default: %(default)s)')
    args = parser.parse_args()
    ds = main_dataset(args)
    assert len(ds) == 1,\
        "Can only create a song from 1 Dataset, try filtering with '-s' or indexind with '-i'"
    np.random.seed(args.seed)
    midi_object = dataset2song(ds, tempo=60/args.bpm)
    out = args.output + (".mid" if args.output.lower()[-4:] != ".mid" else "")
    midi_object.write(out)
