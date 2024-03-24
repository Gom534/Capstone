# [Arduino EMG Sensor Visualizing Project]

![Alt text](img/1.jpg)
![!\[Alt text\](2.jpg)](img/2.jpg)
---

<br>

# 📖 Contents

- [🌈 Background](#-background)
- [🛠 Features](#-features)
- [📈 Architecture](#-Architecture)
- [🪃 Skills](#-skills)
  - [Client](#client-1)
  - [Server](#server-1)
- [🧗 Challenges](#-challenges)
  - [1. 웹소켓 통신 및 백그라운드 데이터 수집](#1-웹소켓-통신-및-백그라운드-데이터-수집)
  - [2. 웹소켓 라이브러리 활용 / 데이터베이스 ORM 사용](#2-웹소켓-라이브러리-활용-/-데이터베이스-ORM-사용)
  - [3. 비동기 처리 및 백그라운드 작업](#3-비동기-처리-및-백그라운드-작업)
  - [4. 실시간 데이터 시각화 및 결과 전송](#3-실시간-데이터-시각화-및-결과-전송)
- [🙏 마무리하며..](#-마무리하며)

<br>

# 🌈 Background
캡스톤디자인 프로젝트는 아두이노를 이용한 빅데이터 기반의 건강관리 시스템이다.
해당 프로젝트 참여 인원은 총 5명이며, 예상 재료비 예산은 약 80만원으로 정하였다. 
아두이노 심전도 센서(Electrocardiogram Sensor)를 이용하여 근전도 생체신호를 모니터 링하고 효율적으로 데이터 관리하여 웹이나 앱을 통해 가시화시켜 시각적으로 알 수 있는 운동 도우미 기능을 만들기 위해 해당 프로젝트를 진행하게 되었다.

<br>

# 🛠 Features
  - Main Page: 실시간 근전도 센서를 통한 데이터 시각화 확인 가능
  - My Page: 데이터를 통한 운동능력 결과확인 가능 
  - Login : 로그인 기능  
<br>

# 📈 Architecture
![Alt text](img/3.jpg)

<br>

# 🪃 Skills

## Client

- HTML, CSS, JavaScript, Python

## Server

- Python
- Flask
- AWS

## Version Control

- Git, Github


<br>

# 🧗 Challenges

기능 개발을 하면서 겪은 어려움 또는 도전은 아래와 같습니다.

<br>

## 1. 웹소켓 통신 및 백그라운드 데이터 수집

웹소켓 통신과 백그라운드 데이터 수집은 비동기적인 작업을 수행하므로 코드의 복잡성이 증가할 수 있습니다.
특히, 백그라운드에서 데이터를 수집하고 동시에 웹소켓을 통해 클라이언트에게 실시간 업데이트를 전달하는 부분은 복잡성이 높을 수 있습니다. 그렇기 때문에 코드를 각 기능에 따라 모듈화하고 함수로 나누면 코드를 이해하기 쉬워집니다. 백그라운드 데이터 수집, 데이터베이스 연동, 모델 학습, 웹소켓 통신 등 각각의 역할을 하는 함수를 작성합니다.
<br>

## 2. 웹소켓 라이브러리 활용 / 데이터베이스 ORM 사용

웹소켓 통신에는 Flask-SocketIO를 사용하고, 클라이언트에게 실시간 업데이트를 전달하는 부분은 해당 라이브러리의 기능을 활용하여 간편하게 구현할 수 있습니다.

데이터베이스 연동에는 Flask-MySQLdb를 사용하고, ORM(Object-Relational Mapping)을 활용하면 SQL 쿼리를 직접 작성하지 않고도 Python 객체로 데이터를 다룰 수 있습니다.
<br>

## 3. 비동기 처리 및 백그라운드 작업

비동기 처리를 위해 asyncio와 async/await를 사용할 수 있습니다. 백그라운드 데이터 수집 및 웹소켓 통신을 비동기로 처리하여 병렬적으로 작업을 수행합니다.
<br>

## 4. 실시간 데이터 시각화 및 결과 전송

실시간으로 데이터를 시각화하고 클라이언트에게 결과를 전달하는 부분은 웹소켓과 함께 구현되어야 합니다.
사용자가 데이터 수집을 시작하면, 해당 데이터를 실시간으로 그래프 등의 형태로 시각화하고 결과를 전송하는 것이 복잡할 수 있습니다. 그렇기 때문에 데이터를 실시간으로 시각화하는 부분에는 Plotly, Matplotlib 등의 라이브러리를 사용하여 그래프를 생성하고 웹 페이지에 표시합니다.

# 🙏 마무리하며...

이 프로젝트를 통하여 팀원간의 협업과 의사소통의 중요성을 알게 되었고 경진대회를 통한 다른 팀들의 여러 작품을 볼 수 있어서 좋은 경험이 되었다.   
