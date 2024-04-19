import math
import datetime
import serial
import random
import csv
import realtime_clssification as rc
import numpy as np

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import StringProperty, ListProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.clock import Clock
from kivy_garden.graph import Graph, LinePlot
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color
from kivy.uix.label import Label

x = [0, 0]
y = [0, 0]
res = 0
cnt = 0

time1 = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

PORT1 = 'COM7'
PORT2 = 57600

Window.size = (360, 640)

gUsername = ""

class LoginScreen(Screen):
    result = StringProperty("")  # resetta risultato

    def authenticate(self, username, password):
        global gUsername
        res = 0
        with open('users.txt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if username == row[0] and password == row[1]:
                    res = 1
                    gUsername = username
                    # Le credenziali sono corrette
                    if row[2] == 'T':
                        App.get_running_app().root.current = 'mainTutor'  # dividi casi user e tutor qui con due screen diversi
                    elif row[2] == 'U':
                        App.get_running_app().root.current = 'mainUser'

        if res == 0:
            self.result = "Wrong username or password"
        else:
            self.result = ""

class SignUpScreen(Screen):
    global user_type
    user_type = "U"
    result = StringProperty("")
    colorUser = ListProperty([0.2, 0.7, 0.2, 1])
    colorTutor = ListProperty([0.8, 0.8, 0.8, 1])

    def select_user_type(self):
        global user_type
        self.colorTutor = [0.8, 0.8, 0.8, 1]
        self.colorUser = [0.2, 0.7, 0.2, 1]
        user_type = "U"

    def select_tutor_type(self):
        global user_type
        self.colorTutor = [0.2, 0.7, 0.2, 1]
        self.colorUser = [0.8, 0.8, 0.8, 1]
        user_type = "T"

    def sign_up(self, new_username, new_password, confirm_password):
        self.result = ""
        if not new_username or not new_username or not confirm_password:
            self.result = "Please fill in all fields"
            return

        elif new_password != confirm_password:
            self.result = 'Passwords do not match'
            return

        # Verifica se l'utente esiste già
        with open('users.txt', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == new_username:
                    self.result = 'Username already exists'
                    return

        with open('users.txt', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([new_username, new_password, user_type])
            self.result = ''

            App.get_running_app().root.current = 'login'

class MainUserScreen(Screen):
    pass

class MainTutorScreen(Screen):
    pass

class HorizontalBarGraph(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wakefulness = 0
        self.fatigue = 0

        # 创建用于显示清醒程度和疲劳程度的标签
        self.label_wakefulness = Label(text="Wakefulness", color=[0, 0, 0, 1], halign='center',
                                       pos=(self.x, self.y + 20))
        self.label_fatigue = Label(text="Fatigue", color=[0, 0, 0, 1], halign='center', pos=(self.x, self.y + 70))

        self.add_widget(self.label_wakefulness)
        self.add_widget(self.label_fatigue)

    def update_data(self, wakefulness, fatigue):
        self.wakefulness = wakefulness
        self.fatigue = fatigue
        self.canvas.before.clear()
        self.draw_bars()

        # 更新标签的位置
        self.label_wakefulness.pos = (
            self.x + self.width / 2 - 60, self.y)
        self.label_fatigue.pos = (
            self.x + self.width / 2 - 60, self.y + 90)

    def draw_bars(self):
        with self.canvas.before:
            # 绘制清醒程度的条形图
            Color(0.2, 0.7, 0.2, 1)  # 绿色
            # 添加一些间隙，避免重叠
            gap = 0
            Rectangle(pos=(self.x, self.y), size=(self.width * (self.wakefulness / 100), self.height))

            # 绘制疲劳程度的条形图
            Color(0.8, 0.2, 0.2, 1)  # 红色
            Rectangle(pos=(self.x, self.y + gap), size=(self.width * (self.fatigue / 100), self.height))

    def draw_bars(self):
        with self.canvas.before:
            # 绘制清醒程度的条形图
            Color(0.2, 0.7, 0.2, 1)  # 绿色
            Rectangle(pos=self.pos, size=(self.width * (self.wakefulness / 100), self.height))

            # 添加一些间隙，避免重叠
            gap = 95
            # 绘制疲劳程度的条形图
            Color(0.8, 0.2, 0.2, 1)  # 红色
            Rectangle(pos=(self.x, self.y + gap), size=(self.width * (self.fatigue / 100), self.height))

class recordScreen(Screen):
    """
    status = StringProperty("Connection in progress...")
    attentionShown = StringProperty("XX")
    meditationShown = StringProperty("XX")
    action = StringProperty("Keep working!")  # aggiungi colore soglia e cambiamento
    colorAction = ListProperty([0.2, 0.7, 0.2, 1])
    """
    value = StringProperty("Fatigue:")
    data = []

    def __init__(self, **kwargs):
        global time1, data
        super().__init__(**kwargs)
        """
        ser = serial.Serial("COM8", 57600)

        if ser.isOpen():
            print("Serial is open")
            # self.status = "Connected"
        """
        """
        with open(time1 + '.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['counter', 'rawValue'])
        """
        with open("TestData.txt", 'r') as f:
            csvreader = csv.reader(f)
            data = [int(row[0]) for row in csvreader]

        graph_theme = {'label_options': {'color': [0, 0, 0, 1], 'bold': True},
                       'tick_color': [0.3, 0.3, 0.3, 1]}

        self.graph = Graph(xlabel='time(s)', x_ticks_minor=5, x_ticks_major=10, y_ticks_major=100,
                           y_grid_label=True, x_grid_label=True, padding=5, x_grid=True, y_grid=True, xmin=0,
                           xmax=10, **graph_theme)  # insert the axis
        self.bar_graph = HorizontalBarGraph(size_hint_x=0.9, size_hint_y=0.1,
                                            pos_hint={'center_x': 0.5, 'center_y': 0.38})

        box1 = self.ids.box1
        box1.add_widget(self.graph)
        box2 = self.ids.box2
        box2.add_widget(self.bar_graph)
        self.plot = LinePlot(line_width=1.3, color=[0, 0.8, 0.3, 1])
        self.plot.points = [(x[i], y[i]) for i in range(len(x))]
        self.graph.add_plot(self.plot)
        self.wakefulness_length = 100
        self.fatigue_length = 0

        Clock.schedule_interval(self.update_data, 0.01)

    def update_data(self, dt):
        global x, y, cnt
        """
        ser = serial.Serial("COM8", 57600)
        
        rawValue = int((ser.readline()).decode())
        """
        # rawValue = random.randint(-200, 200)
        # self.value = "Fatigue: wake"
        """
        with open(time1 + '.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([cnt, rawValue])
        """

        rawValue = data[cnt]

        x.append(cnt)
        y.append(rawValue)
        self.graph.xmax = cnt + 5
        if cnt < 100:
            temp = 0
            space = math.ceil(cnt / 80) * 10
        else:
            temp = cnt - 100
            space = 15
        self.graph.xmin = temp
        self.graph.ymax = max([abs(ele) for ele in y]) + 10  # maybe it gives division by zero error idk why
        self.graph.ymin = -max([abs(ele) for ele in y]) - 10
        self.graph.y_ticks_major = math.ceil((max([abs(ele) for ele in y]) + 10) / 100) * 20
        self.graph.x_ticks_major = space
        cnt += 1
        self.plot.points = [(x[i], y[i]) for i in range(len(x))]
        # 模拟机器学习模型的输出，0表示疲劳，1表示清醒
        if cnt >= rc.sample_rate * 5:
            ml_y = rc.get_realtime_classification(np.array(data[cnt - rc.sample_rate * 5:cnt]), rc.sample_rate)
            ml_output = int(ml_y[0])

            # 处理机器学习模型输出
            if ml_output == 0 and self.fatigue_length < 100:
                self.fatigue_length += 20
                self.wakefulness_length -= 20
                if self.fatigue_length < 60:
                    self.value = "Status: Fatigue"
            elif ml_output == 1 and self.wakefulness_length < 100:
                self.fatigue_length -= 20
                self.wakefulness_length += 20
                if self.fatigue_length < 60:
                    self.value = "Status: Awake"

                # 如果疲劳大于等于60，则发出警告
            # if self.fatigue_length >= 60:
            #    self.value = "Warning!Warning!"

            # 避免长度超过100
            self.wakefulness_length = max(0, min(self.wakefulness_length, 100))
            self.fatigue_length = max(0, min(self.fatigue_length, 100))

            # 更新横向条形图数据
            self.bar_graph.update_data(self.wakefulness_length, self.fatigue_length)

    def stopRecording(self):
        App.get_running_app().root.current = 'mainUser'

class AddUserScreen(Screen):
    result = StringProperty("")
    color = ListProperty([0.2, 0.7, 0.2, 1])

    def verify(self, username, password):
        res = 0
        with open('users.txt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if username == row[0] and password == row[1] and "U" == row[2]:
                    res = 1

                    # logica di aggiunta  write as more argument in the csv
                    with open('users.txt', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "ritrovo nome da file precedente copio e cancello la riga e la ristampo attaccando il nome dell'utente"])

        if res == 0:
            self.color = [0.7, 0.2, 0.2, 1]
            self.result = "Wrong username or password"
        else:
            self.color = [0.2, 0.7, 0.2, 1]
            self.result = "Added correctly"

class RawSignalVisualization(App):
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(SignUpScreen(name='signup'))
        sm.add_widget(MainUserScreen(name='mainUser'))
        sm.add_widget(MainTutorScreen(name='mainTutor'))
        sm.add_widget(AddUserScreen(name='addUser'))
        sm.add_widget(recordScreen(name='record'))
        return sm

if __name__ == '__main__':
    RawSignalVisualization().run()
