# -*- coding: utf-8 -*-
import sys
import struct
from math import * 
from scipy.fftpack import fft, ifft
import numpy as np
from PySide.QtMultimedia import *
from PySide.QtGui import *
from PySide.QtCore import *
from PySide.QtOpenGL import *
from OpenGL.GL import *
from OpenGL.GLU import *


def helz2mel(helz):
    return 1000 * log(helz / 1000. + 1, 2)

def emphasis_filter(data):
    return data[:2] + [data[i] - 0.97 * data[i-1] - 0.2 * data[i-1]
            for i in range(2, len(data))]

class Plotter(QGLWidget):

    def __init__(self,
            plot_range=(-1, 1, -1, 1), 
            islog=False,
            ispolygon=False,
            parent=None):
        super(Plotter, self).__init__(parent)
        self._data = None 

        self.setMinimumSize(1000,300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.plot_range = plot_range
        self.islog = islog
        self.ispolygon = ispolygon
        self.grid_width = 1
        self.plot_width = 2
        self.plot_color = (0.239, 0.619, 1)

    def initializeGL(self):
        self.qglClearColor(Qt.black)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        ortho_range = self.plot_range + (-1, 1)
        glOrtho(*ortho_range)
        self._draw_grid()
        self._draw_data()
       
    def _draw_data(self):
        if not self._data:
            return
        glColor(*self.plot_color)
        if self.ispolygon:
            self._draw_data_polygon()
        else:
            self._draw_data_line()

    def _draw_data_line(self):
        glLineWidth(self.plot_width)
        glBegin(GL_LINE_STRIP)
        for x, y in self._data:
            if self.islog:
                x = 1000 * log(x/1000+1, 2)
            glVertex2f(x, y)
        glEnd()

    def _draw_data_polygon(self):
        glBegin(GL_TRIANGLE_STRIP)
        r = self.plot_range
        glVertex2f(r[0], r[2])
        for i, (x, y) in enumerate(self._data[:-1]):
            x2, _ = self._data[i+1]
            if self.islog:
                x = 1000 * log(x/1000+1, 2)
                x2 = 1000 * log(x2/1000+1, 2)
            glVertex2f(x, y)
            glVertex2f(x2, r[2])
        glVertex2f(*self._data[-1])
        glEnd()

    def _draw_grid(self):
        glLineWidth(self.grid_width)
        glColor(0.5, 0.5, 0.5)
        r = self.plot_range
        xstep = 10 ** max(1, int((log10(r[1])-1)))
        ystep = 10 ** max(1, int((log10(r[3]))))
        glBegin(GL_LINES)
        for x in range(int(r[0]), int(r[1]), xstep):
            glVertex2f(x, r[2])
            glVertex2f(x, r[3])
        for y in range(int(r[2]), int(r[3]), ystep):
            glVertex2f(r[0], y)
            glVertex2f(r[1], y)
        glEnd()
        glColor(1,1,1)

    def setData(self, data):
        if isinstance(data[0], (tuple, list)):
            self._data = data
        else:  # x軸を自動付加
            xs = range(0, len(data))
            self._data = zip(xs, data)
        self.updateGL()


class SlowBar(QProgressBar):
    def __init__(self, parent=None):
        super(SlowBar, self).__init__(parent)
        self._acc = 1

    @Slot()
    def setValue(self, value):
        if self.value() < value:
            filterd_value = value
            self._acc = 1
        else:
            filterd_value = max(self.minimum(),
                    self.value() - self.maximum() * 0.01 * self._acc)
            self._acc += 0.1
        super(SlowBar, self).setValue(filterd_value)
        

class MainWindow(QMainWindow):
    _volumeChanged = Signal(int)
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        #ウィンドウタイトルの設定
        self.setWindowTitle("pySpectra")

        #入力デバイスの取得
        self._input = None
        self._devices = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)

        # UI周りの設定
        self._setup_ui()
        self._setup_toolbar()

        #デフォルト動作オプションの設定
        self.setInputDevice(0)
        self._buffer_combo.setCurrentIndex(2)
        self._set_power_spectle_mode()
        self._mode = 'power_spectle'

        #プロセスタイマー始動
        self._timer = QTimer()
        self._timer.timeout.connect(self._process)
        self._timer.start(0)

    @Slot(int)
    def setInputDevice(self, n):
        if self._input:
            self._input.stop()
        self._input = self._create_audio_input(self._devices[n])
        self._input_buf = self._input.start()
        self._input.stateChanged.connect(self._notify)

    @Slot(int)
    def setBufferSize(self, n):
        self._sample_size = 2 ** (9 + n)
        self._wave_form = [0] * self._sample_size
        self._power_spectle = [0] * self._sample_size
        self._spectle_env = [0] * self._sample_size
        self._wave_viewer.plot_range = (0, self._sample_size, -1, 1)
        # x軸の生成
        df = self._input.format().sampleRate() / float(self._sample_size)
        self._xs = [df * i for i in range(0, self._sample_size / 2)]

    @Slot()
    def _notify(self):
        if self._input.state() is QAudio.IdleState:
            self._input.reset()
            self._input.start()

    @Slot()
    def _process(self):
        if self._freeze_action.isChecked():
            return

        n = self._sample_size
        #入力信号の取得
        try:
            byte = self._input.bytesReady()
            if not byte:
                raise ValueError, "no bytes read"
            wave_form = struct.unpack(
                    str(byte / 2) + "h",
                    self._input_buf.read(byte))
            wave_form = wave_form[:min(n, len(wave_form))]
        except (struct.error, ValueError):
            return

        #ボリュームを求める
        float_form = map(lambda x: x/32768., wave_form)
        abs_form = map(lambda x: abs(x), float_form)
        volume = max(abs_form) * self._bar.maximum()

        #全体波形の更新
        self._wave_form = self._wave_form[len(float_form):] + float_form

        #スペクトルを求める
        ham_window = np.hamming(n)
        if self._emphasis_action.isChecked():
            filterd_form = emphasis_filter(self._wave_form)
        else:
            filterd_form = self._wave_form
        self._power_spectle = np.abs(fft(ham_window * filterd_form))
        self._log_spectle = 10 * np.log10(self._power_spectle + 0.0000001)

        #ケプストラムを求める
        cps = np.real(ifft(self._power_spectle))
        cps_coef = int(self._cepstrum_degree.currentText())
        cps[cps_coef:n - cps_coef + 1] = 0  # 高次を取り除く

        #スペクトル包絡を求める
        self._spectle_env = np.abs(fft(cps, len(self._log_spectle)))
        
        #表示の更新
        self._bar.setValue(volume)
        self._wave_viewer.setData(self._wave_form)
        if self._mode == 'power_spectle':
            self._spectle_viewer.setData(zip(self._xs, self._power_spectle[:n/2]))
        elif self._mode == 'log_spectle':
            self._spectle_viewer.setData(zip(self._xs, self._log_spectle[:n/2]))
        elif self._mode == 'envelope':
            self._spectle_viewer.setData(zip(self._xs, self._spectle_env[:n/2]))

    def _create_audio_input(self, device):
        format = QAudioFormat()
        format.setSampleSize(16)
        format.setChannels(1)
        format.setSampleRate(44100)
        format.setSampleType(QAudioFormat.SignedInt)
        format.setByteOrder(QAudioFormat.LittleEndian)
        format.setCodec("audio/pcm")
        return QAudioInput(device, format)

    def _setup_ui(self):
        #ボリューム表示用のバー
        self._bar = SlowBar()
        self._bar.setOrientation(Qt.Vertical)
        self._bar.setRange(0, 100000)
        self._volumeChanged.connect(self._bar.setValue)

        #ケプストラムの次数調節用コンボボックス
        self._cepstrum_degree = QComboBox()
        for size in range(50, 151, 10):
            self._cepstrum_degree.addItem(str(size))
        self._cepstrum_degree.setCurrentIndex(5)

        #入力信号ビューワー
        self._wave_viewer = Plotter()
        self._wave_viewer.setFixedHeight(100)

        #ケプストラムビューワー
        self._spectle_viewer = Plotter()
        self._spectle_viewer.plot_range = (0, helz2mel(22050), 0, 40)
        self._spectle_viewer.islog = True

        #入力デバイス洗濯用コンボボックス
        self._input_combo = QComboBox()
        for device in self._devices:
            self._input_combo.addItem(device.deviceName())
        self._input_combo.currentIndexChanged.connect(self.setInputDevice)

        #サンプルサイズ選択用コンボボックス
        self._buffer_combo = QComboBox()
        for size in range(5):
            self._buffer_combo.addItem(str(2 ** (9 + size)))
        self._buffer_combo.currentIndexChanged.connect(self.setBufferSize)

        #ビューワーレイアウト
        plotter_lay = QVBoxLayout()
        plotter_lay.addWidget(self._wave_viewer)
        plotter_lay.addWidget(self._spectle_viewer)

        #全体レイアウト
        lay = QHBoxLayout()
        lay.addLayout(plotter_lay)
        lay.addWidget(self._bar)

        w = QWidget()
        w.setLayout(lay)
        self.setCentralWidget(w)


    def _setup_toolbar(self):
        self._toolbar = self.addToolBar("DISP MODE")

        self._log_spectle_action = QAction("LOG SPECTLE", self)
        self._log_spectle_action.setCheckable(True)
        self._log_spectle_action.triggered.connect(self._set_log_spectle_mode)

        self._power_spectle_action = QAction("POWER SPECTLE", self)
        self._power_spectle_action.setCheckable(True)
        self._power_spectle_action.triggered.connect(self._set_power_spectle_mode)

        self._envelope_action = QAction("SPECTLE ENV", self)
        self._envelope_action.setCheckable(True)
        self._envelope_action.triggered.connect(self._set_envelope_mode)

        self._freeze_action = QAction("FREEZE", self)
        self._freeze_action.setCheckable(True)
        self._emphasis_action = QAction("EMPHASIS", self)
        self._emphasis_action.setCheckable(True)
        self._polygon_action = QAction("POLYGON", self)
        self._polygon_action.setCheckable(True)
        self._polygon_action.triggered.connect(self._set_polygon)

        self._toolbar.addAction(self._power_spectle_action)
        self._toolbar.addAction(self._log_spectle_action)
        self._toolbar.addAction(self._envelope_action)

        self._toolbar.addSeparator()

        self._toolbar.addAction(self._freeze_action)
        self._toolbar.addAction(self._emphasis_action)
        self._toolbar.addAction(self._polygon_action)

        self._toolbar.addSeparator()

        self._toolbar.addWidget(QLabel("Input Device"))
        self._toolbar.addWidget(self._input_combo)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(QLabel("Buffer Size"))
        self._toolbar.addWidget(self._buffer_combo)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(QLabel("Cepstrum Degree"))
        self._toolbar.addWidget(self._cepstrum_degree)

    @Slot()
    def _set_polygon(self):
        self._spectle_viewer.ispolygon = self._polygon_action.isChecked()

    @Slot()
    def _set_power_spectle_mode(self):
        n = self._sample_size
        self._mode = 'power_spectle'
        self._power_spectle_action.setChecked(True)
        self._log_spectle_action.setChecked(False)
        self._envelope_action.setChecked(False)
        self._spectle_viewer.plot_range = (0, helz2mel(22050), 0, 40)
        self._spectle_viewer.setData(zip(self._xs, self._power_spectle[:n/2]))

    @Slot()
    def _set_log_spectle_mode(self):
        n = self._sample_size
        self._mode = 'log_spectle'
        self._log_spectle_action.setChecked(True)
        self._power_spectle_action.setChecked(False)
        self._envelope_action.setChecked(False)
        self._spectle_viewer.plot_range = (0, helz2mel(22050), -50, 50)
        self._spectle_viewer.setData(zip(self._xs, self._log_spectle[:n/2]))
    

    @Slot()
    def _set_envelope_mode(self):
        n = self._sample_size
        self._mode = 'envelope'
        self._power_spectle_action.setChecked(False)
        self._log_spectle_action.setChecked(False)
        self._envelope_action.setChecked(True)
        self._spectle_viewer.plot_range = (0, helz2mel(22050), 0, 20)
        self._spectle_viewer.setData(zip(self._xs, self._spectle_env[:n/2]))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    app.exec_()
