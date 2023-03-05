# %%
from logging import root
import re
from flatspin.data import load_output, read_table
from numpy.core.defchararray import count
import pretty_midi
import numpy as np
from warnings import warn
from itertools import count
from collections import defaultdict
import os
from pretty_midi import instrument
import constants

def minmax_norm(arr, min_val=0, max_val=1):
    arr = np.array(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * (max_val - min_val) + min_val
    return arr
def addNotes(notes, scale, instrument, tempo=0.5, durations=None, velocities=None, return_notes=False):
    if return_notes:
        result = [""] * len(notes)
    if durations is not None:
        durations = np.array(durations).astype(int)
    if velocities is not None:
        velocities = minmax_norm(velocities, 0, 127).astype(int)
    for i, note in enumerate(notes):
        if note >= len(scale):
            continue
        note_name = scale[note % len(scale)]
        note_number = pretty_midi.note_name_to_number(note_name)
        end = tempo + tempo*i
        if durations is not None:
            end = tempo*durations[i] + tempo*i
        instrument.notes.append(pretty_midi.Note(
            velocity=100 if velocities is None else velocities[i], pitch=note_number, start=0 + tempo*i, end=end))
        if return_notes:
            result[i] = note_name
    if return_notes:
        return result

def make_intr_with_notes(midi_object, instr_name, scale_octave, notes, scale_tones, tempo, durations, velos, return_notes=False):
    instr_program = pretty_midi.instrument_name_to_program(instr_name)
    instr = pretty_midi.Instrument(program=instr_program)
    scale_tones_transp = octave_transpose_tones(
        scale_tones, scale_octave - 1)
    res = addNotes(notes, scale_tones_transp,
                    instr, tempo, durations, velos, return_notes)
    midi_object.instruments.append(instr)
    if return_notes and not is_dead_instrument(notes):
        return res

def dataset2song(ds, tempo=0.4, scale=None, octave_range=(1, 4), return_notes=False, allowed_instruments=None,
                 durations=None, velocities=None, grid_size=(12, 12), ds_t_filter=None):
    assert 0 <= octave_range[0] < 10 and 0 <= octave_range[1] < 10, "octave range must be between 0 and 10"
    if scale == None:
        scale = make_scale_from_known("major", "C")
    scale_tones = scale2tones(scale)
    instruments_notes, instruments_durs, instruments_velos = dataset2notes(ds, scale_size=len(scale_tones),
                                                                           durations=durations, velocities=velocities,
                                                                            grid_size=grid_size, ds_t_filter=ds_t_filter)
    instruments_durs = instruments_durs if instruments_durs is not None else [None] * len(instruments_notes)
    instruments_velos = instruments_velos if instruments_velos is not None else [None] * len(instruments_notes)
    if allowed_instruments is None:
        allowed_instruments = pretty_midi.constants.INSTRUMENT_MAP
    intruments = np.random.choice(
        allowed_instruments, len(instruments_notes), replace=len(instruments_notes) > len(allowed_instruments))
    octaves = np.random.choice(
        np.arange(*octave_range), len(instruments_notes), replace=True)
    midi_object = pretty_midi.PrettyMIDI()
    if return_notes:
        score_info = {}
    for i_name, scale_octave, i_notes, i_durs, i_velos in zip(intruments, octaves, instruments_notes, instruments_durs, instruments_velos):
        res = make_intr_with_notes(midi_object, i_name, scale_octave, i_notes, scale_tones, tempo, i_durs, i_velos, return_notes)
        if res:
            score_info[i_name] = res
    if return_notes:
        return midi_object, score_info
    return midi_object

def splits(arr, n) :
    """discretize array into n bins and return the bin index for each element
        if not enough unique values in arr, bins are biased towards left (smaller values)
        """
    return np.digitize(arr, (np.arange(n) * np.ptp(arr))/n + np.min(arr)) - 1

def dataset2notes(ds, grid_size=(12, 12), scale_size=8, durations=None, velocities=None, ds_t_filter=None):
    geom = read_table(ds.tablefile("geom"))
    poss = np.array(list(zip(geom["posx"].values, geom["posy"].values)))
    #grid = Grid.fixed_grid(np.array(poss), grid_size)
    mag = load_output(ds, "mag", grid_size=grid_size, flatten=False, t=ds_t_filter)

    angle = vector_colors(mag[..., 0], mag[..., 1])  # [0,2pi)
    
    norm_angle = angle/(2*np.pi)  # [0,1)
    norm_angle *= scale_size  # [0,scale_size)
    norm_angle = norm_angle.round().astype(int)
    norm_angle[norm_angle >= scale_size] = 0
    instr_notes = norm_angle.T.reshape(np.prod(grid_size), -1)
    if durations is not None or velocities is not None:
        if velocities is None:
            instr_velo = np.ones_like(instr_notes)
        else:
            velocities = velocities if velocities is not None else [1]
            magnitudes = np.linalg.norm(mag, axis=-1)
            instr_velo = np.array(velocities)[splits(magnitudes, len(velocities))]            
            instr_velo = instr_velo.T.reshape(np.prod(grid_size), -1) * 100
        if durations is None:
            durations = np.ones_like(instr_notes)
        else:
            angle_diff = np.abs(np.diff(angle, axis=0, prepend=0)) % (2*np.pi)
            angle_diff[angle_diff > np.pi] = 2*np.pi - angle_diff[angle_diff > np.pi] # shortest angle difference
            instr_dur = np.array(durations)[splits(angle_diff, len(durations))]
            instr_dur = instr_dur.T.reshape(np.prod(grid_size), -1)
        assert instr_velo.shape == instr_dur.shape == instr_notes.shape
        return [filter_repeat_notes(i_n, scale_size) for i_n in instr_notes], instr_dur, instr_velo

    return [filter_repeat_notes(i_n, scale_size) for i_n in instr_notes], None, None


def filter_repeat_notes(notes, rest_note_num=8):
    """replace consecutive duplicates with a special rest note value"""
    notes = notes.copy()
    edif = np.concatenate(([1], np.ediff1d(notes)))
    notes[edif == 0] = rest_note_num
    return notes


def is_dead_instrument(notes, rest_note_num=8, start_from=1):
    """an instrument is dead if it plays no notes after 'start_from'"""
    return all(notes[start_from:] == rest_note_num)


def vector_colors(U, V):
    C = np.arctan2(V, U)  # color
    C[C < 0] = 2*np.pi + C[C < 0]
    return C





def num2note(num):
    return _NOTES[num]


def note2num(note):
    num = _NOTE2NUM[note[0].upper()]
    # check sharp/flat
    num += 1 if note[-1] == "#" else (-1 if note[-1] == "b" else 0)
    return num


def is_note(note):
    if type(note) is not str:
        return False
    if len(note) > 1:
        if len(note) > 2 or note[1] not in ("#", "b"):
            return False
        note = note[0]
    return 97 <= ord(note.lower()) <= 103


def add_semitones(start_note, semitones=1):
    """returns the note n semitones after start_note where n=semitones"""
    return num2note((note2num(start_note) + semitones) % 12)


def make_scale_from_known(scale_name, root_note):
    scale = [root_note]
    intervals = KNOWN_SCALES[scale_name]
    for intv in intervals:
        scale.append(add_semitones(scale[-1], intv))
    return scale


def parse_custom_scale(scale_text):
    scale_notes = scale_text.split(",")
    if not all(map(is_note, scale_notes)):
        raise ValueError(f"Unable to parse custom scale: '{scale_text}'")
    return scale_notes


def parse_scale(scale_text):
    scale_text = scale_text.strip("'"" ")
    if "," in scale_text:
        return parse_custom_scale(scale_text)

    scale_text = scale_text.split("_")
    if scale_text[-1] not in KNOWN_SCALES:
        raise ValueError(f"unknown scale: '{scale_text[-1]}'")

    root_note = scale_text[0] if len(scale_text) == 2 else "C"
    if not is_note(root_note):
        warn(f"root note '{root_note}' not recognised, defaulting to 'C'")
        root_note = 'C'
    return make_scale_from_known(scale_text[-1], root_note)


def normalise_note(note):
    """ensures notes are in consistant form"""
    return num2note(note2num(note))


def scale2tones(scale, start_octave=1):
    """makes scale starting at start_octave"""
    scale = list(map(normalise_note, scale))
    octave = start_octave
    tones = []
    last_note_num = -1

    for note in scale:
        note_num = note2num(note)
        if note_num <= last_note_num:
            octave += 1
        tones.append(note + str(octave))
        last_note_num = note_num
    return tones


def octave_transpose_tones(tones, ammount=1):
    return [tone[:-1] + str(int(tone[-1])+ammount) for tone in tones]


def tones2lilypond(tones, note_length=1/4):
    """note length as fraction of a semibreve"""
    return [tone2lilypond(tone, note_length) for tone in tones]


def tone2lilypond(tone, note_length=1/4):
    """note length as fraction of a semibreve"""
    length = round(1/note_length)
    if tone == "":
        return f"r{length}"
    # replace # b with lilypond sharp/flat symbol
    tone = tone[0] + tone[1:].replace("#", "s").replace("b", "f")

    tone = tone.lower()
    # calculate the number of ,/' to specify octave in lilypond
    oct_num = int(tone[-1])
    num_commas = 3 - oct_num
    num_apos = oct_num - 3
    # remove old pitch number and add comas/apostrophes
    tone = tone[:-1] + num_commas * "," + num_apos * "'"
    return tone+str(length)


def group_rests(tones, beats_per_bar=4):
    rest_count = 0
    beat_count = beats_per_bar
    for i in reversed(range(len(tones))):
        if tones[i][0] != "r":
            if rest_count == 0:  # no rests do nothing
                continue
            else:  # hit a non-rest, group current rests
                tones = tones[:i+1] + (["r4", "r2"] if rest_count ==
                                       3 else [f"r{round(4/rest_count)}"]) + tones[i+rest_count+1:]
                rest_count = 0
        else:
            rest_count += 1
            if rest_count == 4:  # max 4 rests can begrouped
                tones = tones[:i+1] + (["r4", "r2"] if rest_count ==
                                       3 else [f"r{round(4/rest_count)}"]) + tones[i+rest_count+1:]
                rest_count = 0
            elif beat_count == 1:  # if on 1st beat of bar end rest group
                tones = tones[:i+1] + (["r4", "r2"] if rest_count ==
                                       3 else [f"r{round(4/rest_count)}"]) + tones[i+rest_count+1:]
                rest_count = 0
                beat_count = beats_per_bar + 1
        beat_count -= 1
    return tones


def score_info2lilypond_file(score_info):
    staffs = []
    for name in score_info:
        voice = abjad.Voice(
            " ".join(group_rests(tones2lilypond(score_info[name]))), name=name
        )
        start_markup = abjad.StartMarkup(markup=abjad.Markup(name))
        abjad.attach(start_markup, voice[0])
        staffs.append(abjad.Staff([voice], name=f"{name}_Staff"))

    score = abjad.Score(staffs, name="Score")

    return abjad.LilyPondFile(items=[score])


def save_lilypond_pdf(lilyfile, outdir):
    abjad.show(lilyfile, output_directory=outdir,
               should_open=False, should_persist_log=False)

# known scales given by number of semitolnes between each note
KNOWN_SCALES = {"major":        [2, 2, 1, 2, 2, 2, 1],
                "minor":        [2, 1, 2, 2, 1, 3, 1],
                "chromatic":    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
_NOTE2NUM = {"A": 0, "B": 2, "C": 3, "D": 5, "E": 7, "F": 8, "G": 10}

KNOWN_INSTRUMENT_CLASSES = {"all": pretty_midi.constants.INSTRUMENT_MAP,
                            "synth" : constants.SYNTH_ONLY,
                            "no_sfx" : constants.NO_SFX,
                            "rock" : constants.ROCK,
                            "voice" : constants.VOICE,
                            }
# %%
if __name__ == '__main__':
    from flatspin.cmdline import main_dataset_argparser, main_dataset, parse_time

    parser = main_dataset_argparser("magnets2midi", True)
    parser.add_argument('-t', default='::1',help='time slice to use slice dataset. default: %(default)s')
    parser.add_argument('--bpm', type=float, default=200,
                        help='beats per minute (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to use (default: %(default)s)')
    parser.add_argument("--scale", default="major", help="""
    known scales can be specified eg: major, G#_minor, Bb_chromatic,...
    known scales: [major, minor, chromatic], (deaults to C if no root note given)

    Alternatively, custom scales can be input as: C,D,E,F#,G#,Bb,C
    where notes will be interpretted to be in accending order.
    """)
    parser.add_argument("--instruments",type=str,  default="all",help="""
    specify a subset of the instruments to use. 
    Current choices:
    all, synth, no_sfx, rock
    """)
    parser.add_argument('--oct-range', type=int, nargs=2, default=[1, 4], help="octave ranges (min max) that the instruments can choose from")
    parser.add_argument('--grid-size', type=int, nargs="+", default=[4, 4], help="""
    grid size to use for the song either as a single number (4) (will be used for both dimensions)
      or as a list of 2 numbers (4 4)""")
    parser.add_argument('--durations', type=float, nargs="+", default=None,
                        help="list of possible note durations. (normal=1)")
    parser.add_argument('--velocities', type=float, nargs="+", default=None,
                        help="list of possible note velocities. (normal=1), (will be multiplied by 100)")
    parser.add_argument('--pdf', action='store_true',
                        help='save music score as pdf')

    args = parser.parse_args()
    scale = parse_scale(args.scale)
    ds = main_dataset(args)
    assert len(args.grid_size) <= 2, f"grid size must be 1 or 2 numbers, got {len(args.grid_size)}"
    if len(args.grid_size) == 1:
        args.grid_size = [args.grid_size[0], args.grid_size[0]]
    assert len(ds) == 1, \
        "Can only create a song from 1 Dataset, try filtering with '-s' or indexing with '-i'"
    t = parse_time(args.t)
    np.random.seed(args.seed)
    instruments =[]
    if args.instruments.lower() not in KNOWN_INSTRUMENT_CLASSES:
        warn(f"unkown instrument class: {args.instruments}, defaulting to all")
        instruments = KNOWN_INSTRUMENT_CLASSES["all"]
    else:
        instruments = KNOWN_INSTRUMENT_CLASSES[args.instruments.lower()]


    if args.pdf:
        import abjad
        midi_object, score_info = dataset2song(
            ds, tempo=60/args.bpm, scale=scale, return_notes=True, allowed_instruments=instruments,
            durations=args.durations, velocities=args.velocities, oct_range=tuple(args.oct_range),
            grid_size=tuple(args.grid_size), ds_t_filter=t)
    else:
        midi_object = dataset2song(ds, tempo=60/args.bpm, scale=scale, allowed_instruments=instruments,
                                   durations=args.durations, velocities=args.velocities, octave_range=tuple(args.oct_range),
                                      grid_size=tuple(args.grid_size), ds_t_filter=t)
    out = args.output + (".mid" if args.output.lower()[-4:] != ".mid" else "")
    midi_object.write(out)

    if args.pdf:
        print("midi done, creating score pdf...")
        lilyfile = score_info2lilypond_file(score_info)
        outdir = os.getcwd()
        save_lilypond_pdf(lilyfile, outdir)
