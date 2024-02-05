import pathlib

try:
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
except ModuleNotFoundError:
    rank_zero_info = print

from utils.hparams import hparams

_initialized = False
_ALL_CONSONANTS_SET = set()
_ALL_VOWELS_SET = set()
_dictionary = {
    'AP': ['AP'],
    'SP': ['SP']
}
_phoneme_list: list


def locate_dictionary():
    """
    Search and locate the dictionary file.
    Order:
    1. hparams['dictionary']
    2. hparams['g2p_dictionary']
    3. 'dictionary.txt' in hparams['work_dir']
    4. file with same name as hparams['g2p_dictionary'] in hparams['work_dir']
    :return: pathlib.Path of the dictionary file
    """
    assert 'dictionary' in hparams or 'g2p_dictionary' in hparams, \
        'Please specify a dictionary file in your config.'
    config_dict_path = pathlib.Path(hparams.get('dictionary', hparams.get('g2p_dictionary')))
    if config_dict_path.exists():
        return config_dict_path
    work_dir = pathlib.Path(hparams['work_dir'])
    ckpt_dict_path = work_dir / config_dict_path.name
    if ckpt_dict_path.exists():
        return ckpt_dict_path
    ckpt_dict_path = work_dir / 'dictionary.yaml'
    if ckpt_dict_path.exists():
        return ckpt_dict_path
    ckpt_dict_path = work_dir / 'dictionary.txt'
    if ckpt_dict_path.exists():
        return ckpt_dict_path
    raise FileNotFoundError('Unable to locate the dictionary file. '
                            'Please specify the right dictionary in your config.')


def _build_dict_and_list(dictionary_path: pathlib.Path):
    global _dictionary, _phoneme_list

    _set = set()
    with open(dictionary_path, 'r', encoding='utf8') as _df:
        _lines = _df.readlines()
    for _line in _lines:
        _pinyin, _ph_str = _line.strip().split('\t')
        _dictionary[_pinyin] = _ph_str.split()
    for _list in _dictionary.values():
        [_set.add(ph) for ph in _list]
    _phoneme_list = sorted(list(_set))
    rank_zero_info('| load phoneme set: ' + str(_phoneme_list))


def _initialize_consonants_and_vowels():
    # Currently we only support two-part consonant-vowel phoneme systems.
    for _ph_list in _dictionary.values():
        _ph_count = len(_ph_list)
        if _ph_count == 0 or _ph_list[0] in ['AP', 'SP']:
            continue
        elif len(_ph_list) == 1:
            _ALL_VOWELS_SET.add(_ph_list[0])
        else:
            _ALL_CONSONANTS_SET.add(_ph_list[0])
            _ALL_VOWELS_SET.add(_ph_list[1])

def _load_txt_dictionary(dictionary_path: pathlib.Path):
    _build_dict_and_list(dictionary_path)
    _initialize_consonants_and_vowels()

def _load_yaml_dictionary(dictionary_path: pathlib.Path):
    """
    load openutau-style yaml dictionary
    """
    global _dictionary, _phoneme_list
    import yaml
    with open(dictionary_path, 'r', encoding='utf8') as f:
        _dict = yaml.safe_load(f)
    _dictionary = {entry["grapheme"]:entry["phonemes"] for entry in _dict['entries']}
    _phoneme_list = []
    for symbol in sorted(_dict['symbols'], key=lambda x: x['symbol']):
        _phoneme_list.append(symbol["symbol"])
        if(symbol["symbol"] in ['AP', 'SP']):
            continue
        if(symbol["type"]=="vowel"):
            _ALL_VOWELS_SET.add(symbol["symbol"])
        else:
            _ALL_CONSONANTS_SET.add(symbol["symbol"])
    #validate
    for k, v in _dictionary.items():
        if not isinstance(v, list):
            raise ValueError(f'Invalid entry for {k}: {v}')
        for ph in v:
            if ph not in _phoneme_list:
                raise ValueError(f'Invalid phoneme {ph} in entry for {k}')

def _load_dictionary():
    dictionary_path = locate_dictionary()
    if dictionary_path.suffix == '.yaml':
        _load_yaml_dictionary(dictionary_path)
    else:
        _load_txt_dictionary(dictionary_path)

def _initialize():
    global _initialized
    if not _initialized:
        _load_dictionary()
        _initialized = True


def get_all_consonants():
    _initialize()
    return sorted(_ALL_CONSONANTS_SET)


def get_all_vowels():
    _initialize()
    return sorted(_ALL_VOWELS_SET)


def build_dictionary() -> dict:
    _initialize()
    return _dictionary


def build_phoneme_list() -> list:
    _initialize()
    return _phoneme_list
