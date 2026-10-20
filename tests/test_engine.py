import numpy as np
import pytest
import soundfile as sf
from engine.analyzer import analyze
from engine.mastering import master
from engine.exporter import export_audio
from engine.dynamics import opto_compress, vocal_rider

RATE=24000
def tone(freq, seconds=1, amp=.2):
    t=np.arange(int(RATE*seconds))/RATE
    x=(amp*np.sin(2*np.pi*freq*t)).astype(np.float32)
    return np.column_stack([x,x*.9])

@pytest.mark.parametrize('preset',['Afrobeat Warm','Afrobeat Punchy','R&B Smooth','Vocal Clear','Rap & Hip-Hop','Pop Bright','Natural Master','Custom'])
def test_presets_return_valid_bounded_audio(preset):
    x=tone(440); y,warnings=master(x,RATE,{'bass':.3},preset)
    assert y.shape==x.shape and np.isfinite(y).all() and np.max(np.abs(y))<1

def test_mono_stereo_duration_preserved(tmp_path):
    for channels in [1,2]:
        x=tone(440)
        if channels==1:x=x[:,0]
        src=tmp_path/f'in{channels}.wav'; dst=tmp_path/f'out{channels}.wav'
        sf.write(src,x,RATE); audio,rate,info=analyze(str(src)); y,_=master(audio,rate,info['bands'],'Natural Master');export_audio(str(dst),y,rate)
        actual,actual_rate=sf.read(dst,always_2d=True)
        assert actual_rate==RATE and actual.shape==(len(x),channels)

def test_limiter_protects_hot_input():
    x=tone(1000,amp=1.2);y,_=master(x,RATE,{'bass':.2},'Natural Master')
    assert np.max(np.abs(y))<=10**(-1/20)+1e-6

def test_silence_is_finite():
    y,_=master(np.zeros((RATE,2),np.float32),RATE,{'bass':0},'Natural Master')
    assert np.isfinite(y).all() and not np.any(y)

def test_bass_heavy_material_gets_gentle_low_frequency_reduction():
    bass=tone(90,seconds=2,amp=.55)
    info=analyze_signal(bass)
    y,_=master(bass,RATE,info,'Afrobeat Warm')
    assert np.sqrt(np.mean(y*y)) < np.sqrt(np.mean(bass*bass))

def test_balanced_material_is_not_needlessly_louder_or_clipped():
    t=np.arange(RATE)/RATE
    x=np.column_stack([.12*np.sin(2*np.pi*440*t)+.08*np.sin(2*np.pi*4000*t),.12*np.sin(2*np.pi*440*t)+.08*np.sin(2*np.pi*4000*t)]).astype(np.float32)
    y,_=master(x,RATE,{'bass':.25},'Natural Master')
    assert np.max(np.abs(y))<=10**(-1/20)+1e-6
    assert np.sqrt(np.mean(y*y)) <= np.sqrt(np.mean(x*x))*1.05
    assert abs(analyze_signal_loudness(y)-analyze_signal_loudness(x))<1.5

def test_default_mastering_is_level_matched_and_audibly_processes_mix():
    t=np.arange(RATE*2)/RATE
    mono=(.22*np.sin(2*np.pi*90*t)+.14*np.sin(2*np.pi*1000*t)+.08*np.sin(2*np.pi*9000*t))
    x=np.column_stack([mono,mono*.92]).astype(np.float32)
    original_loudness=analyze_signal_loudness(x)
    bands=analyze_signal(x)

    y,_=master(x,RATE,bands,'Afrobeat Warm')
    mastered_loudness=analyze_signal_loudness(y)

    assert np.isfinite(y).all()
    assert np.max(np.abs(y))<=10**(-1/20)+1e-6
    assert abs(mastered_loudness-original_loudness)<1.5
    assert np.sqrt(np.mean((y-x)**2))>0.01

def test_quiet_audio_stays_finite_and_bounded():
    x=tone(440,amp=.001);y,_=master(x,RATE,{'bass':.3},'Natural Master')
    assert np.isfinite(y).all() and np.max(np.abs(y))<.01

def test_vocal_rider_reduces_large_level_jumps_without_changing_stereo_balance():
    quiet=np.full((RATE,2),.04,dtype=np.float32)
    loud=np.full((RATE,2),.4,dtype=np.float32)
    x=np.concatenate([quiet,loud,quiet])

    y=vocal_rider(x,RATE)
    levels=[np.sqrt(np.mean(y[start:start+RATE]**2)) for start in (0,RATE,2*RATE)]

    assert np.isfinite(y).all()
    assert max(levels)/min(levels)<4.5
    assert np.allclose(y[:,0],y[:,1],atol=1e-6)
    assert np.max(np.abs(y))<=np.max(np.abs(x))*1.7

def test_opto_compressor_controls_sustained_level_smoothly():
    x=tone(440,seconds=2,amp=.5)

    y=opto_compress(x,RATE)

    assert np.isfinite(y).all()
    assert np.sqrt(np.mean(y*y))<np.sqrt(np.mean(x*x))
    assert np.max(np.abs(np.diff(y,axis=0)))<np.max(np.abs(np.diff(x,axis=0)))*1.1

def test_long_signal_completes_without_changing_length():
    x=tone(330,seconds=12);y,_=master(x,RATE,{'bass':.3},'Natural Master')
    assert len(y)==len(x) and y.shape[1]==2

def test_invalid_file_fails(tmp_path):
    bad=tmp_path/'bad.wav';bad.write_text('not audio')
    with pytest.raises(Exception): analyze(str(bad))

def test_analyze_action_returns_audio_info(tmp_path):
    from engine.main import run
    source=tmp_path/'import.wav'
    sf.write(source,tone(440),RATE)

    result=run({'action':'analyze','input':str(source)})

    assert result['name']=='import.wav'
    assert result['path']==str(source.resolve())
    assert 'input' not in result

def test_preview_action_converts_flac_to_playable_wav(tmp_path):
    from engine.main import run
    source=tmp_path/'import.flac'
    preview=tmp_path/'preview.wav'
    sf.write(source,tone(440),RATE)

    result=run({'action':'preview','input':str(source),'output':str(preview)})

    audio,rate=sf.read(preview,always_2d=True)
    assert result['path']==str(preview.resolve())
    assert rate==RATE
    assert audio.shape==(RATE,2)

def test_master_action_writes_output_file(tmp_path):
    from engine.main import run
    source=tmp_path/'source.wav'
    output=tmp_path/'master.wav'
    sf.write(source,tone(440),RATE)

    result=run({'action':'master','input':str(source),'output':str(output),'preset':'Natural Master'})

    assert result['output_path']==str(output.resolve())
    assert output.is_file()
    mastered,rate=sf.read(output,always_2d=True)
    assert rate==RATE
    assert mastered.shape==(RATE,2)

def test_export_refuses_original_overwrite(tmp_path):
    p=tmp_path/'x.wav';sf.write(p,tone(440),RATE)
    with pytest.raises(ValueError):
        # The safety check is applied by the pipeline entry point.
        from engine.main import run
        import json
        run({'action':'master','input':str(p),'output':str(p),'preset':'Natural Master'})

def analyze_signal(audio):
    from engine.analyzer import _db
    mono=audio.mean(axis=1); spectrum=np.abs(np.fft.rfft(mono))**2; freqs=np.fft.rfftfreq(len(mono),1/RATE)
    total=max(float(spectrum.sum()),1e-20)
    return {'bass':float(spectrum[freqs<250].sum()/total),'mids':float(spectrum[(freqs>=250)&(freqs<4000)].sum()/total),'highs':float(spectrum[freqs>=4000].sum()/total)}

def analyze_signal_loudness(audio):
    from engine.loudness import integrated_loudness
    return integrated_loudness(audio.astype(np.float64),RATE)
