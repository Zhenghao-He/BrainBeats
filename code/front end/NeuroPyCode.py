import math
import random
import csv

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import StringProperty, ListProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.clock import Clock
from kivy_garden.graph import Graph, LinePlot

x = [0]
y = [0]
res = 0
cnt = 0
onPage = 0

Window.size = (360, 640)

gUsername = ""


class LoginScreen(Screen):
    result = StringProperty("")

    def authenticate(self, username, password):
        global gUsername
        res = 0
        with open('users.txt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if username == row[0] and password == row[1]:
                    res = 1
                    # Le credenziali sono corrette
                    if row[2] == 'T':
                        gUsername = row[3]
                        App.get_running_app().root.current = 'mainTutor'
                    elif row[2] == 'U':
                        gUsername = username
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
        global TUsername, TPsw
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

        if user_type == "T":
            TUsername = new_username
            TPsw = new_password
            App.get_running_app().root.current = 'addUser'
        else:
            with open('users.txt', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([new_username, new_password, user_type])
                self.result = ''

            App.get_running_app().root.current = 'login'


class MainUserScreen(Screen):

    def startRecording(self):  # usa onscreen
        global onPage
        onPage = 1
        App.get_running_app().root.current = 'record'

    pass


class MainTutorScreen(Screen):
    pass


class recordScreen(Screen):
    status = StringProperty("Connection in progress...")
    attentionShown = StringProperty("XX")
    meditationShown = StringProperty("XX")
    action = StringProperty("Keep working!")  # aggiungi colore soglia e cambiamento
    colorAction = ListProperty([0.2, 0.7, 0.2, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with open('data.csv', 'r') as f:
            self.status = "Connected"

        graph_theme = {'label_options': {'color': [0, 0, 0, 1], 'bold': True},
                       'tick_color': [0.3, 0.3, 0.3, 1]}

        self.graph = Graph(xlabel='time(s)', x_ticks_minor=5, x_ticks_major=10, y_ticks_major=50,
                           y_grid_label=True, x_grid_label=True, padding=5, x_grid=True, y_grid=True, xmin=0,
                           xmax=10, **graph_theme)

        box = self.ids.box
        box.add_widget(self.graph)
        self.plot = LinePlot(line_width=1.5, color=[0, 0.8, 0.3, 1])
        self.plot.points = [(x[i], y[i]) for i in range(len(x))]
        self.graph.add_plot(self.plot)

        Clock.schedule_interval(self.update_data, 1)  # inserire in on screen

    def update_data(self, dt):
        global x, y, cnt
        if onPage == 1:
            with open('data.csv', 'r') as f:
                csvreader = csv.reader(f)
                data = [row for row in csvreader]
                self.attentionShown = str((int(data[-1][0])))
                self.meditationShown = str((int(data[-1][1])))

                rawValue = (int(data[-1][2]))

            x.append(cnt)
            y.append(rawValue)
            self.graph.xmax = cnt + 5
            self.graph.ymax = max([abs(ele) for ele in y]) + 10  # maybe it gives division by zero error idk why
            self.graph.ymin = -max([abs(ele) for ele in y]) - 10
            self.graph.x_ticks_major = math.ceil(cnt / 100) * 10
            cnt += 1
            self.plot.points = [(x[i], y[i]) for i in range(len(x))]

    def stopRecording(self):
        global onPage
        onPage = 0
        App.get_running_app().root.current = 'mainUser'
        with open('data.csv', 'r') as f:
            csvreader = csv.reader(f)
            data = [row for row in csvreader]
        with open('lastData-' + gUsername + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(row for row in data)


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
                    with open('users.txt', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([TUsername, TPsw, "T", username])

        if res == 0:
            self.color = [0.7, 0.2, 0.2, 1]
            self.result = "Wrong username or password"
        else:
            self.color = [0.2, 0.7, 0.2, 1]
            self.result = "Added correctly"


class viewScreen(Screen):
    result = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        y = [0]
        x = [0]

        graph_theme = {'label_options': {'color': [0, 0, 0, 1], 'bold': True},
                       'tick_color': [0.3, 0.3, 0.3, 1]}

        self.graph3 = Graph(xlabel='time(s)', ylabel='RawData', x_ticks_minor=5, x_ticks_major=10, y_ticks_major=50,
                            y_grid_label=True, x_grid_label=True, padding=5, x_grid=True, y_grid=True, xmin=0,
                            xmax=10, **graph_theme)

        self.graph1 = Graph(xlabel='time(s)', ylabel='Attention', x_ticks_minor=5, x_ticks_major=10, y_ticks_major=50,
                            y_grid_label=True, x_grid_label=True, padding=5, x_grid=True, y_grid=True, xmin=0,
                            xmax=10, **graph_theme)

        self.graph2 = Graph(xlabel='time(s)', ylabel='Meditation', x_ticks_minor=5, x_ticks_major=10, y_ticks_major=50,
                            y_grid_label=True, x_grid_label=True, padding=5, x_grid=True, y_grid=True, xmin=0,
                            xmax=10, **graph_theme)

        box1 = self.ids.box1
        box1.add_widget(self.graph1)
        self.plot1 = LinePlot(line_width=1.5, color=[0, 0.8, 0.3, 1])
        self.graph1.add_plot(self.plot1)

        box2 = self.ids.box2
        box2.add_widget(self.graph2)
        self.plot2 = LinePlot(line_width=1.5, color=[0, 0.8, 0.3, 1])
        self.graph2.add_plot(self.plot2)

        box3 = self.ids.box3
        box3.add_widget(self.graph3)
        self.plot3 = LinePlot(line_width=0.5, color=[0, 0.8, 0.3, 1])
        self.graph3.add_plot(self.plot3)

    def readData(self):
        global result

        y = [0]
        x = [0]
        data = []

        try:
            with open('lastData-' + gUsername + '.csv', 'r') as f:
                csvreader = csv.reader(f)
                data = [row for row in csvreader]
                self.result=""
        except IOError:
            self.result = "No recording available"

        for el in data[1:][:]:
            att = int(el[0])
            y.append(att)

        x = [el for el in range(len(y))]

        self.graph1.xmax = len(y)
        self.graph1.y_ticks_major = math.ceil((max([abs(ele) for ele in y]) + 10) / 100) * 20
        self.graph1.x_ticks_major = math.ceil(len(y) / 50) * 10

        self.plot1.points = [(x[i], y[i]) for i in range(len(x))]

        x = [0]
        y = [0]

        for el in data[1:][:]:
            att = int(el[1])
            y.append(att)

        x = [el for el in range(len(y))]

        self.graph2.xmax = len(y)
        self.graph2.y_ticks_major = math.ceil((max([abs(ele) for ele in y]) + 10) / 100) * 20
        self.graph2.x_ticks_major = math.ceil(len(y) / 50) * 10

        self.plot2.points = [(x[i], y[i]) for i in range(len(x))]

        x = [0]
        y = [0]

        for el in data[1:][:]:
            att = int(el[2])
            y.append(att)

        x = [el for el in range(len(y))]

        self.graph3.xmax = len(y)
        self.graph3.ymax = max([abs(ele) for ele in y]) + 10  # maybe it gives division by zero error idk why
        self.graph3.ymin = -max([abs(ele) for ele in y]) - 10
        self.graph3.y_ticks_major = math.ceil((max([abs(ele) for ele in y]) + 10) / 50) * 20
        self.graph3.x_ticks_major = math.ceil(len(y) / 50) * 10

        self.plot3.points = [(x[i], y[i]) for i in range(len(x))]


class viewTutorScreen(Screen):
    result = StringProperty("")

    def __init__(self, **kwargs):  # need to initialize when open the page not at the start of app
        super().__init__(**kwargs)
        y = [0]
        x = [0]

        graph_theme = {'label_options': {'color': [0, 0, 0, 1], 'bold': True},
                       'tick_color': [0.3, 0.3, 0.3, 1]}

        self.graph3 = Graph(xlabel='time(s)', ylabel='RawData', x_ticks_minor=5, x_ticks_major=10, y_ticks_major=50,
                            y_grid_label=True, x_grid_label=True, padding=5, x_grid=True, y_grid=True, xmin=0,
                            xmax=10, **graph_theme)

        self.graph1 = Graph(xlabel='time(s)', ylabel='Attention', x_ticks_minor=5, x_ticks_major=10, y_ticks_major=50,
                            y_grid_label=True, x_grid_label=True, padding=5, x_grid=True, y_grid=True, xmin=0,
                            xmax=10, **graph_theme)

        self.graph2 = Graph(xlabel='time(s)', ylabel='Meditation', x_ticks_minor=5, x_ticks_major=10, y_ticks_major=50,
                            y_grid_label=True, x_grid_label=True, padding=5, x_grid=True, y_grid=True, xmin=0,
                            xmax=10, **graph_theme)

        box1 = self.ids.box1
        box1.add_widget(self.graph1)
        self.plot1 = LinePlot(line_width=1.5, color=[0, 0.8, 0.3, 1])
        self.graph1.add_plot(self.plot1)

        box2 = self.ids.box2
        box2.add_widget(self.graph2)
        self.plot2 = LinePlot(line_width=1.5, color=[0, 0.8, 0.3, 1])
        self.graph2.add_plot(self.plot2)

        box3 = self.ids.box3
        box3.add_widget(self.graph3)
        self.plot3 = LinePlot(line_width=0.5, color=[0, 0.8, 0.3, 1])
        self.graph3.add_plot(self.plot3)

    def readData(self):
        global result

        data = []
        y = [0]
        x = [0]

        try:
            with open('lastData-' + gUsername + '.csv', 'r') as f:
                csvreader = csv.reader(f)
                data = [row for row in csvreader]
                self.result=""
        except IOError:
            self.result = "No recording available"

        for el in data[1:][:]:
            att = int(el[0])
            y.append(att)

        x = [el for el in range(len(y))]

        self.graph1.xmax = len(y)
        self.graph1.y_ticks_major = math.ceil((max([abs(ele) for ele in y]) + 10) / 100) * 20
        self.graph1.x_ticks_major = math.ceil(len(y) / 50) * 10

        self.plot1.points = [(x[i], y[i]) for i in range(len(x))]

        x = [0]
        y = [0]

        for el in data[1:][:]:
            att = int(el[1])
            y.append(att)

        x = [el for el in range(len(y))]

        self.graph2.xmax = len(y)
        self.graph2.y_ticks_major = math.ceil((max([abs(ele) for ele in y]) + 10) / 100) * 20
        self.graph2.x_ticks_major = math.ceil(len(y) / 50) * 10

        self.plot2.points = [(x[i], y[i]) for i in range(len(x))]

        x = [0]
        y = [0]

        for el in data[1:][:]:
            att = int(el[2])
            y.append(att)

        x = [el for el in range(len(y))]

        self.graph3.xmax = len(y)
        self.graph3.ymax = max([abs(ele) for ele in y]) + 10  # maybe it gives division by zero error idk why
        self.graph3.ymin = -max([abs(ele) for ele in y]) - 10
        self.graph3.y_ticks_major = math.ceil((max([abs(ele) for ele in y]) + 10) / 50) * 20
        self.graph3.x_ticks_major = math.ceil(len(y) / 50) * 10

        self.plot3.points = [(x[i], y[i]) for i in range(len(x))]


class NeuroPyCode(App):
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(SignUpScreen(name='signup'))
        sm.add_widget(MainUserScreen(name='mainUser'))
        sm.add_widget(MainTutorScreen(name='mainTutor'))
        sm.add_widget(AddUserScreen(name='addUser'))
        sm.add_widget(recordScreen(name='record'))
        sm.add_widget(viewScreen(name='view'))
        sm.add_widget(viewTutorScreen(name='viewTutor'))
        return sm


if __name__ == '__main__':
    NeuroPyCode().run()
