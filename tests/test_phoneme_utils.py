import pathlib
import tempfile
import unittest

from utils.hparams import hparams
from utils.phoneme_utils import PhonemeDictionary, load_phoneme_dictionary


class ExtraPhonemesTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = pathlib.Path(directory.name)
        self.ja = self.root / 'ja.txt'
        self.en = self.root / 'en.txt'
        self.ja.write_text('a\ta\n', encoding='utf8')
        self.en.write_text('a\ta\n', encoding='utf8')

    def test_single_language_accepts_full_and_short_extra_phonemes(self):
        dictionary = PhonemeDictionary({'ja': self.ja}, extra_phonemes=['EP', 'ja/cl'])

        for sentence in ['a cl EP', 'ja/a ja/cl EP']:
            for lang in [None, 'ja']:
                with self.subTest(sentence=sentence, lang=lang):
                    tokens = dictionary.encode(sentence, lang=lang)
                    self.assertEqual(dictionary.decode(tokens), 'a cl EP')
        self.assertEqual(dictionary.vocab_size, 6)  # AP, EP, SP, a, cl, and padding.

    def test_single_language_extra_phoneme_can_be_merged(self):
        dictionary = PhonemeDictionary(
            {'ja': self.ja}, extra_phonemes=['ja/cl'], merged_groups=[['ja/cl', 'SP']]
        )

        self.assertEqual(dictionary.encode('cl ja/cl'), dictionary.encode('SP SP'))
        self.assertEqual(dictionary.vocab_size, 4)

    def test_extra_phoneme_already_in_dictionary_has_one_id(self):
        original = PhonemeDictionary({'ja': self.ja})
        dictionary = PhonemeDictionary({'ja': self.ja}, extra_phonemes=['ja/a'])

        self.assertEqual(dictionary.vocab_size, original.vocab_size)
        self.assertEqual(dictionary.encode('a ja/a'), original.encode('a ja/a'))

    def test_config_loader_resolves_single_language_extra_phonemes(self):
        previous = hparams.copy()
        try:
            hparams.clear()
            hparams.update({
                'work_dir': str(self.root),
                'dictionaries': {'ja': str(self.ja)},
                'extra_phonemes': ['EP', 'ja/cl'],
                'merged_phoneme_groups': [['ja/cl', 'SP']],
            })
            dictionary = load_phoneme_dictionary()

            self.assertEqual(dictionary.encode('cl EP', lang='ja'), dictionary.encode('SP EP'))
        finally:
            hparams.clear()
            hparams.update(previous)

    def test_multilingual_extra_phonemes_keep_language_prefixes(self):
        dictionary = PhonemeDictionary(
            {'ja': self.ja, 'en': self.en}, extra_phonemes=['EP', 'ja/cl', 'en/cl']
        )

        ja = dictionary.encode_one('cl', lang='ja')
        en = dictionary.encode_one('cl', lang='en')
        self.assertNotEqual(ja, en)
        self.assertEqual(dictionary.decode([ja, en]), 'ja/cl en/cl')
        self.assertEqual(dictionary.encode('ja/cl en/cl'), [ja, en])
        self.assertEqual(dictionary.encode('EP', lang='ja'), dictionary.encode('EP', lang='en'))

    def test_extra_phoneme_validation_is_preserved(self):
        for extra, message in [('en/cl', 'unrecognized language'), ('ja/SP', 'conflicts')]:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                PhonemeDictionary({'ja': self.ja}, extra_phonemes=[extra])


if __name__ == '__main__':
    unittest.main()
