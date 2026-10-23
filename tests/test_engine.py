import numpy as np
import pytest
import soundfile as sf
from engine.analyzer import analyze
from engine.mastering import master
from engine.exporter import export_audio
from engine.dynamics import opto_compress, program_leveler, vocal_rider
from engine.limiter import limit
from engine.filters import highpass
from engine.true_peak import true_peak
from engine import vocal

RATE=24000
def tone(freq, seconds=1, amp=.2):
    t=np.arange(int(RATE*seconds))/RATE
    x=(amp*np.sin(2*np.pi*freq*t)).astype(np.float32)
    return np.column_stack([x,x*.9])

@pytest.mark.parametrize('preset',['AUTO','Balanced','Afrobeat','R&B','Vocal Clarity','Warm & Smooth','Powerful','AI Music Polish','Afrobeat Warm','Afrobeat Punchy','R&B Smooth','Vocal Clear','Rap & Hip-Hop','Pop Bright','Natural Master','Afrobeat Vocal Studio','Custom'])
def test_presets_return_valid_bounded_audio(preset):
    x=tone(440); y,warnings=master(x,RATE,{'bass':.3},preset)
    assert y.shape==x.shape and np.isfinite(y).all() and np.max(np.abs(y))<1

def test_auto_preset_selects_a_profile_from_analyzed_bands():
    _,warnings=master(tone(90),RATE,{'bass':.65,'low_mids':.15,'highs':.2},'AUTO')

    assert any('Profil AUTO sélectionné : Afrobeat.' in warning for warning in warnings)

def test_afrobeat_vocal_studio_keeps_level_and_true_peak_controlled():
    t=np.arange(RATE*3)/RATE
    envelope=np.repeat([.12,.42,.18],RATE)
    signal=(envelope*(.7*np.sin(2*np.pi*220*t)+.3*np.sin(2*np.pi*2400*t)))
    audio=np.column_stack([signal,signal*.94]).astype(np.float32)

    mastered,_=master(audio,RATE,{'bass':.3,'low_mids':.2,'highs':.25},
                      'Afrobeat Vocal Studio')

    assert mastered.shape==audio.shape
    assert np.isfinite(mastered).all()
    assert np.max(np.abs(mastered))<=10**(-1/20)+1e-6
    assert abs(analyze_signal_loudness(mastered)-analyze_signal_loudness(audio))<1.5

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

def test_true_peak_limiter_controls_inter_sample_peaks():
    rate=44100
    t=np.arange(rate*2)/rate
    x=np.column_stack([
        .95*np.sin(2*np.pi*11025*t+np.pi/4),
        .95*np.sin(2*np.pi*11025*t+np.pi/4),
    ]).astype(np.float32)

    y=limit(x,-1.0)

    assert np.max(np.abs(y))<1
    assert true_peak(y)<=10**(-1.05/20)+1e-6


def test_limiter_recovers_after_loud_transient_instead_of_latching_gain_reduction():
    t=np.arange(RATE*2)/RATE
    quiet=.2*np.sin(2*np.pi*440*t)
    loud=1.5*np.sin(2*np.pi*440*t)
    audio=np.column_stack([
        np.concatenate([loud[:RATE//2],quiet[RATE//2:]]),
        np.concatenate([loud[:RATE//2],quiet[RATE//2:]]),
    ]).astype(np.float32)

    processed=limit(audio,-1.0,sample_rate=RATE)
    recovered=processed[RATE:RATE*2]

    assert np.sqrt(np.mean(recovered**2)) > .12
    assert np.max(np.abs(processed)) <= 10**(-1/20)+1e-6


def test_analysis_reports_true_peak_and_crest_factor(tmp_path):
    source=tmp_path/'analysis.wav'
    sf.write(source,tone(440),RATE)

    _,_,info=analyze(str(source))

    assert info['true_peak_dbfs']>=info['peak_dbfs']-0.1
    assert info['crest_factor_db']>=0
    assert {'bass','low_mids','mids','highs'}<=info['bands'].keys()

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
    assert levels[0]<=np.sqrt(np.mean(quiet**2))*1.01
    assert levels[2]<=np.sqrt(np.mean(quiet**2))*1.01
    assert levels[1]<np.sqrt(np.mean(loud**2))
    assert max(levels)/min(levels)>5
    assert np.allclose(y[:,0],y[:,1],atol=1e-6)
    assert np.max(np.abs(y))<=np.max(np.abs(x))*1.01

def test_vocal_rider_changes_gain_slowly_to_avoid_pumping():
    t=np.arange(RATE*4)/RATE
    levels=np.tile(np.repeat([.1,.5],RATE//2),4)
    signal=np.sin(2*np.pi*440*t)*levels
    x=np.column_stack([signal,signal]).astype(np.float32)

    y=vocal_rider(x,RATE)
    block_size=RATE//20
    rms=np.array([np.sqrt(np.mean(y[i:i+block_size]**2))
                  for i in range(0,len(y),block_size)])
    gain_db=20*np.log10(np.maximum(rms,1e-8)/np.maximum(
        np.array([np.sqrt(np.mean(x[i:i+block_size]**2))
                  for i in range(0,len(x),block_size)]),1e-8))

    assert np.max(np.abs(np.diff(gain_db)))<0.35


def test_program_leveler_narrows_section_level_changes_without_pumping():
    t=np.arange(RATE*6)/RATE
    levels=np.repeat([.06,.38,.08,.42,.07,.35],RATE)
    signal=levels*np.sin(2*np.pi*220*t)
    audio=np.column_stack([signal,signal*.95]).astype(np.float32)

    processed=program_leveler(audio,RATE)
    block_size=RATE//20
    input_rms=np.array([
        np.sqrt(np.mean(audio[start:start+RATE]**2))
        for start in range(0,len(audio),RATE)
    ])
    output_rms=np.array([
        np.sqrt(np.mean(processed[start:start+RATE]**2))
        for start in range(0,len(processed),RATE)
    ])

    assert np.isfinite(processed).all()
    assert np.ptp(20*np.log10(output_rms)) < np.ptp(20*np.log10(input_rms))-1.5
    gain_db=20*np.log10(np.maximum(
        np.array([
            np.sqrt(np.mean(processed[i:i+block_size]**2))
            for i in range(0,len(processed),block_size)
        ]),1e-8)/np.maximum(
        np.array([
            np.sqrt(np.mean(audio[i:i+block_size]**2))
            for i in range(0,len(audio),block_size)
        ]),1e-8))
    assert np.max(np.abs(np.diff(gain_db)))<0.4


def test_opto_compressor_controls_sustained_level_smoothly():
    x=tone(440,seconds=2,amp=.5)

    y=opto_compress(x,RATE)

    assert np.isfinite(y).all()
    assert np.sqrt(np.mean(y*y))<np.sqrt(np.mean(x*x))
    assert np.max(np.abs(np.diff(y,axis=0)))<np.max(np.abs(np.diff(x,axis=0)))*1.1


def test_vocal_clarity_master_preset_increases_presence_automatically():
    t=np.arange(RATE*2)/RATE
    voice=np.column_stack([
        .12*np.sin(2*np.pi*300*t)+.08*np.sin(2*np.pi*3200*t),
        .11*np.sin(2*np.pi*300*t)+.07*np.sin(2*np.pi*3200*t),
    ]).astype(np.float32)
    bands={'bass':0.0,'low_mids':0.1,'mids':0.8,'highs':0.1}
    neutral,_=master(voice,RATE,bands,'Natural Master')
    clear,_=master(voice,RATE,bands,'Vocal Clarity')
    frequencies=np.fft.rfftfreq(len(voice),1/RATE)
    neutral_spectrum=np.abs(np.fft.rfft(neutral.mean(axis=1)))
    clear_spectrum=np.abs(np.fft.rfft(clear.mean(axis=1)))
    low=neutral_spectrum[np.argmin(np.abs(frequencies-300))]
    presence=neutral_spectrum[np.argmin(np.abs(frequencies-3200))]
    clear_low=clear_spectrum[np.argmin(np.abs(frequencies-300))]
    clear_presence=clear_spectrum[np.argmin(np.abs(frequencies-3200))]

    assert clear.shape==voice.shape and np.isfinite(clear).all()
    assert clear_presence/clear_low > presence/low


def test_vocal_highpass_reduces_rumble_without_cutting_vocal_body():
    t=np.arange(RATE*2)/RATE
    audio=np.column_stack([
        .5*np.sin(2*np.pi*45*t)+.2*np.sin(2*np.pi*220*t),
        .5*np.sin(2*np.pi*45*t)+.2*np.sin(2*np.pi*220*t),
    ]).astype(np.float32)

    filtered=highpass(audio,RATE,75)
    low_ratio=np.sqrt(np.mean(filtered[:,0]**2))/np.sqrt(np.mean(audio[:,0]**2))
    body_ratio=np.sqrt(np.mean(
        highpass(np.column_stack([.2*np.sin(2*np.pi*220*t)]*2),RATE,75)**2
    ))/np.sqrt(np.mean((.2*np.sin(2*np.pi*220*t))**2))

    assert low_ratio<0.65
    assert body_ratio>0.9


def test_automatic_vocal_mastering_corrects_boosts_and_preserves_mono(monkeypatch):
    vocals=tone(440,amp=.05)
    instrumental=tone(110,amp=.2)
    monkeypatch.setattr(
        vocal,'_separate',
        lambda audio,rate,progress:(vocals,instrumental,rate),
    )
    monkeypatch.setattr(vocal,'_estimate_key',lambda stem,rate:(9,'La mineur'))
    corrected=[]

    def correct(stem,rate,root,mode,progress):
        corrected.append((root,mode))
        return stem

    monkeypatch.setattr(vocal,'_correct_pitch',correct)
    source=(vocals+instrumental).mean(axis=1,keepdims=True)
    processed,warnings=vocal.enhance_vocals(source,RATE)

    assert corrected==[(9,'mineur')]
    assert processed.shape==source.shape and np.isfinite(processed).all()
    assert np.sqrt(np.mean(processed**2))>np.sqrt(np.mean(source**2))
    assert any('La mineur' in warning for warning in warnings)


def test_pitch_correction_moves_off_note_vocal_toward_its_scale():
    t=np.arange(RATE*2)/RATE
    off_note=np.column_stack([
        .2*np.sin(2*np.pi*230*t),
        .2*np.sin(2*np.pi*230*t),
    ]).astype(np.float32)

    corrected=vocal._correct_pitch(off_note,RATE,9,'mineur')
    segment=corrected[RATE//4:RATE*7//4].mean(axis=1)
    spectrum=np.abs(np.fft.rfft(segment))
    frequencies=np.fft.rfftfreq(len(segment),1/RATE)
    corrected_pitch=frequencies[np.argmax(spectrum)]

    assert np.isfinite(corrected).all()
    assert abs(corrected_pitch-220)<abs(230-220)


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

def test_master_action_writes_output_file(tmp_path,monkeypatch):
    from engine.main import run
    monkeypatch.setattr(
        'engine.main.enhance_vocals',
        lambda audio,rate,progress:(audio,['Correction vocale de test.']),
    )
    source=tmp_path/'source.wav'
    output=tmp_path/'master.wav'
    sf.write(source,tone(440),RATE)

    result=run({
        'action':'master','input':str(source),'output':str(output),
        'preset':'Afrobeat Vocal Studio',
    })

    assert result['output_path']==str(output.resolve())
    assert output.is_file()
    mastered,rate=sf.read(output,always_2d=True)
    assert rate==RATE
    assert mastered.shape==(RATE,2)
    assert result['report']['loudness_change_lu'] is not None
    assert result['preset']=='Afrobeat Vocal Studio'
    assert 'Correction vocale de test.' in result['warnings']
    assert result['output']['true_peak_dbfs']<=-1.05
    assert result['report']['true_peak_change_db']==pytest.approx(
        result['output']['true_peak_dbfs']-result['input']['true_peak_dbfs'])

def test_standard_master_skips_automatic_vocal_separation(tmp_path,monkeypatch):
    from engine.main import run
    monkeypatch.setattr(
        'engine.main.enhance_vocals',
        lambda *args: pytest.fail('Vocal separation should require its dedicated preset.'),
    )
    source=tmp_path/'source.wav'
    output=tmp_path/'master.wav'
    sf.write(source,tone(440),RATE)

    result=run({
        'action':'master','input':str(source),'output':str(output),
    })

    assert output.is_file()
    assert result['preset']=='AUTO'
    assert any('Profil AUTO sélectionné :' in warning
               for warning in result['warnings'])
    assert not any('Correction vocale automatique' in warning
                   for warning in result['warnings'])

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
