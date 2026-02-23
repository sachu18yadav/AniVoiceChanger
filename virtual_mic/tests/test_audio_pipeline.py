import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from virtual_mic import VoiceChangerEngine

def test_voice_changer_engine_init(mocker):
    # Mocking hardware dependencies
    mocker.patch('virtual_mic.sd', autospec=True)
    
    engine = VoiceChangerEngine(sample_rate=48000, block_size=512)
    
    assert engine.sample_rate == 48000
    assert engine.block_size == 512
    assert engine.worker_thread is None
    assert engine.current_effect == "passthrough"

def test_voice_changer_process_passthrough(mocker):
    mocker.patch('virtual_mic.sd', autospec=True)
    
    engine = VoiceChangerEngine()
    
    # Create fake 512-sample float32 audio block
    test_block = np.random.rand(512, 1).astype(np.float32)
    
    processed = engine._process_block(test_block, "passthrough")
    
    # Passthrough should return identical array
    np.testing.assert_array_equal(processed, test_block)

def test_voice_changer_dsp_pitch_shift(mocker):
    mocker.patch('virtual_mic.sd', autospec=True)
    
    engine = VoiceChangerEngine(sample_rate=48000)
    engine.semitones = 12
    
    # Sine wave block
    t = np.linspace(0, 512/48000, 512, endpoint=False)
    test_block = (np.sin(2 * np.pi * 440 * t)).reshape(-1, 1).astype(np.float32)
    
    # Process it with anime_girl (DSP)
    # Since pedalboard might not be installed in the test env, we spy on the method
    # to ensure the engine correctly routes the block to the DSP processor.
    spy = mocker.spy(engine.anime_voice, 'process')
    processed = engine._process_block(test_block, "anime_girl")
    
    spy.assert_called_once()
    
    # Same shape
    assert processed.shape == test_block.shape
