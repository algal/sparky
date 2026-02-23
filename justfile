#: list targets
default:
	@just --list

#: test
test:
        PYTHONPATH=. .venv/bin/python -m pytest tests/ -q

#: test_vad_capture
testvad:
        PYTHONPATH=. .venv/bin/python -m pytest tests/test_vad_capture.py -q

#: compile native extensions (run after switching branches)
compile:
        cd sparky_mvp/core && g++ -shared -fPIC -o libwebrtc_apm_shim.so apm_shim.cpp \
                $(pkg-config --cflags --libs webrtc-audio-processing-1)
        @echo "compiled: sparky_mvp/core/libwebrtc_apm_shim.so"

#: run
run:
        PYTHONPATH=. .venv/bin/python main.py
