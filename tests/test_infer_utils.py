import unittest

import librosa

from utils.infer_utils import trans_key


class TransKeyTest(unittest.TestCase):
    def test_preserves_note_cents(self):
        for note, key, expected in [
            ('A4+25', 2, 'B4+25'),
            ('C4-24', -2, 'A#3-24'),
            ('A#3+40', 12, 'A#4+40'),
            ('B4-38', -12, 'B3-38'),
        ]:
            with self.subTest(note=note, key=key):
                segments = [{'note_seq': f'{note} rest', 'f0_seq': '0 0'}]

                result = trans_key(segments, key)

                self.assertIs(result, segments)
                self.assertEqual(result[0]['note_seq'], f'{expected} rest')

    def test_keeps_transposed_notes_aligned_with_f0(self):
        # A4 raised by 25 cents, followed by an unvoiced frame.
        f0 = 440 * 2 ** (25 / 1200)
        segment = {
            'note_seq': 'A4+25 rest',
            'f0_seq': f'{f0} 0',
            'f0_timestep': '0.005',
            'ph_seq': 'a SP',
            'ph_dur': '0.5 0.1',
        }

        trans_key([segment], 2)

        shifted_f0, unvoiced_f0 = map(float, segment['f0_seq'].split())
        note_f0 = librosa.note_to_hz(segment['note_seq'].split()[0], round_midi=False)
        self.assertAlmostEqual(shifted_f0, note_f0, delta=0.05)
        self.assertEqual(unvoiced_f0, 0)
        self.assertEqual(segment['f0_timestep'], '0.005')
        self.assertEqual(segment['ph_seq'], 'a SP')
        self.assertEqual(segment['ph_dur'], '0.5 0.1')

    def test_preserves_integer_note_format(self):
        segment = {'note_seq': 'C4 C#4 rest Bb3', 'f0_seq': '0'}

        trans_key([segment], 2)

        self.assertEqual(segment['note_seq'], 'D4 D#4 rest C4')


if __name__ == '__main__':
    unittest.main()
