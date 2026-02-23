import pytest
import os
import sys

# Add virtual_mic to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_state import AppState

def test_app_state_initialization(mocker):
    # Mock SessionManager.load_session to return a predictable dict
    mocker.patch('utils.session_manager.SessionManager.load_session', return_value={
        "backend_mode": "Local DSP",
        "last_mode": "test_profile",
        "last_pitch": 5,
        "ptt_enabled": True,
        "ptt_hotkey": "<shift>"
    })
    
    state = AppState()
    
    # Assert initialized values match the mocked session load
    assert state.backend_mode == "Local DSP"
    assert state.profile == "test_profile"
    assert state.pitch == 5
    assert state.ptt_enabled is True
    assert state.ptt_hotkey == "<shift>"
    
    # Assert default component states
    assert state.local_test_active is False
    assert state.local_record_active is False
    assert state.passthrough_active is False
    assert state.ptt_active is False

def test_app_state_update_and_subscribe(mocker):
    mocker.patch('utils.session_manager.SessionManager.load_session', return_value={})
    state = AppState()
    
    # Track if callback was triggered
    callback_triggered = False
    received_val = None
    
    def my_callback(key, val):
        nonlocal callback_triggered, received_val
        if key == "pitch":
            callback_triggered = True
            received_val = val
            
    state.subscribe(my_callback)
    
    # Change pitch and verify callback runs
    state.update("pitch", 16)
    
    assert state.pitch == 16
    assert callback_triggered is True
    assert received_val == 16
