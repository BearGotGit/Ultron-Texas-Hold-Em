import pytest

from connect.adapters import to_treys_card, normalize_action, game_view_from_server


def test_to_treys_card_with_string():
    c = to_treys_card('As')
    assert isinstance(c, int)
    assert c != 0


def test_to_treys_card_with_dict():
    card_dict = {'rank': 14, 'suit': 's'}
    c = to_treys_card(card_dict)
    assert isinstance(c, int)
    assert c != 0


def test_to_treys_card_with_int_and_none():
    assert to_treys_card(123) == 123
    assert to_treys_card(None) == 0


def test_to_treys_card_bad_dict_returns_zero():
    assert to_treys_card({'rank': None}) == 0


def test_normalize_action_variants():
    assert normalize_action('CALL') == 'call'
    assert normalize_action('Check') == 'call'
    assert normalize_action('fold') == 'fold'
    assert normalize_action('Bet') == 'raise'
    assert normalize_action('') == 'call'


def test_game_view_from_server_basic():
    msg = {
        'state': {
            'table': {
                'players': [
                    {'id': 'p1', 'stack': 100},
                    {'id': 'me', 'stack': 200},
                    None,
                ]
            }
        }
    }
    gv = game_view_from_server(msg, player_id='me')
    assert gv['my_seat'] == 1
    assert gv['self_player']['id'] == 'me'
