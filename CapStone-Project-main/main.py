from flask import Flask, render_template, request, redirect, url_for, session, g
from flask_mysqldb import MySQL
from flask_socketio import SocketIO
import serial
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
app = Flask(__name__)

app.secret_key = '1234'
# MySQL 설정
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'user'

mysql = MySQL(app)
socketio = SocketIO(app)

count = 1
dumbbell_weight = 0
control = {'run': False}
# ardu = serial.Serial('COM4', 1000000)

@app.before_request
def before_request():
    g.logged_in = session.get('logged_in', False)
    if g.logged_in:
        g.user_id = session.get('user_id')
        print(g.user_id)
    else:
        g.user_id = None


@app.route('/')
def index():
    return render_template('main.html', logged_in=g.logged_in, user_id=g.user_id)


@app.route('/loginhtml')
def login_page():
    return render_template('login.html')


@app.route('/joinhtml')
def join_page():
    return render_template('join_membership.html')


@app.route('/mypagehtml')
def mypage_page():
    if g.user_id:
        return render_template('mypage.html', user_id=g.user_id)
    else:
        return redirect(url_for('login_page'))


@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    # MySQL 연결
    cur = mysql.connection.cursor()
    # 회원 정보 저장
    cur.execute("INSERT INTO users (id, passwd) VALUES (%s, %s)", (username, password))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('index'))


@app.route('/login', methods=['POST'])
def login(): 
    global control
    username = request.form['username']
    password = request.form['password']
    # MySQL 연결
    cur = mysql.connection.cursor()
    # 사용자 인증
    cur.execute("SELECT id,passwd FROM users WHERE id = %s AND passwd = %s", (username, password))
    user = cur.fetchone()
    if user[0]==username and user[1]==password:
        # 로그인 성공
        session['logged_in'] = True
        session['user_id'] = user[0]
        cur.close()
        return "Login successful"
    else:
        # 로그인 실패
        session['logged_in'] = False
        session['user_id'] = None
        cur.close()
        return redirect(url_for('login_page'))


@app.route('/logout')
def logout():
    # 세션 만료
    session.pop('logged_in', False)
    session.pop('user_id', None)
    return redirect(url_for('index'))


@app.route('/startDataCollection', methods=['POST'])
def start_data_collection():
    global dumbbell_weight, control
    dumbbell_weight = request.form.get('dumbbellWeight')
    control['run'] = True  # 데이터 수집 중지를 위해 'run' 값을 False로 설정합니다.
    update_global_control()
    handle_serial_data(g.user_id)
    return 'Data collection start'

# 아두이노 데이터 수집 중지를 처리하는 라우트
@app.route('/stopDataCollection', methods=['POST'])
def stop_data_collection():
    global dumbbell_weight, control
    control['run'] = False  # 데이터 수집 중지를 위해 'run' 값을 False로 설정합니다.
    update_global_control()
    return 'Data collection stopped'

def save_data_to_mysql(count, user_id ,first_data, second_data): 
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO arduino (count, user_id, first_data, second_data, dumbbel, times) VALUES (%s, %s,%s, %s, %s, CURRENT_TIMESTAMP)",(count, user_id ,first_data, second_data, dumbbell_weight))
    mysql.connection.commit()
    cur.close()

def calculate_rms(data):
    data = [float(x) for x in data]  # 문자열을 숫자로 변환
    squared_data = [x ** 2 for x in data]
    mean_squared = sum(squared_data) / len(squared_data)
    rms = math.sqrt(mean_squared)
    return rms

def update_global_control():
    global control,dumbbell_weight
    dumbbell_weight = dumbbell_weight  # 변경된 값을 다시 전역 변수에 할당
    control = control  # 변경된 값을 다시 전역 변수에 할당
    print(control, dumbbell_weight)

def total_analyze_data(dumbbell_weight):
    # Fetch data from MySQL
    cur = mysql.connection.cursor()
    cur.execute("SELECT first_data, second_data FROM arduino WHERE dumbbel =%s", (dumbbell_weight))
    data = cur.fetchall()
    if len(data) > 0:
        x1 = [float(row[0]) for row in data]
        x2 = [float(row[1]) for row in data]

        channel1_rms = calculate_rms(x1)
        channel2_rms = calculate_rms(x2)
        Average_rms = [channel1_rms, channel2_rms]
        avg_rms =calculate_rms(Average_rms)
    return avg_rms

def trainmodel(dumbbell_weight):
    # Fetch data from MySQL
    cur = mysql.connection.cursor()
    cur.execute("SELECT user_id, first_data, second_data FROM arduino WHERE dumbbel = %s", (dumbbell_weight,))
    data = cur.fetchall()
    user_rms_values = {}
    if len(data) > 0:
        user_ids = set([row[0] for row in data])
        for user_id in user_ids:
            user_data = [(row[1], row[2]) for row in data if row[0] == user_id]
            x1 = []
            x2 = []
            for row in user_data:
                x1_value = float(row[0])
                x2_value = float(row[1])
                x1.append(x1_value)
                x2.append(x2_value)
            channel1_rms = calculate_rms(x1)
            channel2_rms = calculate_rms(x2)
            Average_rms = [channel1_rms, channel2_rms]
            avg_rms = calculate_rms(Average_rms)
            user_rms_values[user_id] = avg_rms

        X = np.array([[float(row[1]), float(row[2])] for row in data])
        y = np.array([float(user_rms_values[row[0]])  for row in data])
        # 데이터 분할
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Linear Regression 모델 학습
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 모델 평가
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = model.score(X_test, y_test)
        print("RMSE:", rmse)
        print("R-squared:", r2)

    return model

@app.route('/self', methods=['POST'])
def self_analyze_data():
    global dumbbell_weight
    # Fetch data from MySQL
    cur = mysql.connection.cursor()
    cur.execute("SELECT first_data, second_data FROM arduino WHERE user_id =%s AND dumbbel =%s", (g.user_id, dumbbell_weight))
    data = cur.fetchall()

    if len(data) > 0:
        x1 = [float(row[0]) for row in data]
        x2 = [float(row[1]) for row in data]

        channel1_rms = calculate_rms(x1)
        channel2_rms = calculate_rms(x2)

        X = np.array([float(channel1_rms),float(channel2_rms)], dtype=np.float64).reshape(1, -1)
        trained_model =trainmodel(dumbbell_weight)
        total_avg_rms=total_analyze_data(dumbbell_weight)
        predictions = trained_model.predict(X)
        print(dumbbell_weight + "kg" + " 본인의 값: " + str(predictions[0]))
        print(dumbbell_weight + "kg" + " 전체 평균값: " + str(total_avg_rms))
        socketio.emit('result', {'predicted_weight': predictions[0],"total_avg_rms": total_avg_rms, "dumbbell_weight": dumbbell_weight, "user_id": g.user_id})
    return 'Success'
        
def handle_serial_data(user_id):
    global count, ardu, control,dumbbell_weight
    if control['run']:
        while True:
            data = ardu.readline().decode('cp949').strip()
            try:
                first_data, second_data = map(float, filter(None, data.split(',')))
                print(f"count: {count}, first_data: {first_data}, second_data: {second_data}")
                save_data_to_mysql(count, user_id,first_data, second_data)
                count += 1
                # Emit data to connected clients
                socketio.emit('message', {'first_data': first_data, 'second_data': second_data})
                if not control['run']:      
                    if count > 1:
                        count = 1
                        break
            except ValueError as e:
                print(f"ValueError: {e}")

if __name__ == '__main__':
    socketio.run(app, port=5500)
    
        
