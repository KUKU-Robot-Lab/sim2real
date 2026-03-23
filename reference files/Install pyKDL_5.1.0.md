# Isaac Sim 5.1.0

## 목표

- `orocos_kinematics_dynamics`의 `python_orocos_kdl`(PyKDL)을 **Isaac Sim 5.1 내부 Python(3.11)** 기준으로 빌드하여 `.so` 생성
- 빌드 후 Isaac Sim의 `python.sh`로 `import PyKDL` 성공 확인
- (중요) Python 3.11에서 터지는 **구버전 pybind11 / 시스템 pybind11 경로** 문제 방지

---

## 0) 전제 및 변수 설정

```bash
# Isaac Sim 설치 경로
export ISAAC_SIM=~/isaacsim-5.1.0
[export ISAAC_SIM=/home/user/isaacsim/5.1.0]

export ISAAC_PY=$ISAAC_SIM/python.sh

# 버전 확인 (Python 3.11이어야 함)
$ISAAC_PY -c "import sys; print(sys.version); print(sys.executable)"
```

> 사용자께서 쓰시는 alias가 있다면, 실제로는 `python.sh`를 실행하는 형태로 쓰는 걸 권장
> 
> 
> 예: `alias isaacpy=~/isaacsim-5.1.0/python.sh`
> 

---

## 1) Isaac Python(3.11) 환경에 빌드 툴 / pybind11 설치

### 1-1. pip/빌드 기본 도구 업데이트

```bash
$ISAAC_PY -m pip install -U pip setuptools wheel
```

> 주의:
> 
> Isaac Sim 5.1.0 번들 Python 환경에서는 위 커맨드로 전역 업그레이드를 하면 기존 패키지 제약이 깨질 수 있습니다.
> 실제로 다음과 같은 충돌이 발생할 수 있습니다.
> 
> - `isaaclab-rl`은 `packaging<24` 필요
> - 일부 `nvidia-srl-*` 패키지는 `lxml<5` 필요
> - `wheel 0.46.x`는 `packaging>=24`를 요구
> 
> 따라서 이미 충돌이 발생했다면, `pyKDL` 빌드 기준으로 아래처럼 안정화한 뒤 진행하는 편이 안전합니다.
> 
> ```bash
> $ISAAC_PY -m pip install --force-reinstall "packaging<24" "wheel<0.46" "lxml>=4.9.2,<5.0.0"
> ```
> 
> `pyKDL` 빌드만 목적이라면 `pip/setuptools/wheel`을 굳이 다시 올리지 않고, `pybind11`만 정상 설치되어 있어도 다음 단계 진행이 가능합니다.

### 1-2. pybind11 설치 (Python 3.11 호환 버전)

```bash
$ISAAC_PY -m pip install -U "pybind11>=2.10.4"
```

### 1-3. pybind11 정상 설치 확인 (중요)

```bash
$ISAAC_PY -c "import pybind11; print('pybind11 ver:', pybind11.__version__); print('pybind11 file:', pybind11.__file__)"
$ISAAC_PY -m pybind11 --includes
$ISAAC_PY -m pybind11 --cmakedir
```

**정상 기준**

- `pybind11.__file__`이 `.../site-packages/pybind11/...` 아래
- `python -m pybind11 --includes`가 `I...pybind11/include ...` 형태로 출력

---

## 2) 소스 다운로드

```bash
cd ~/Downloads
git clone https://github.com/orocos/orocos_kinematics_dynamics.git
cd orocos_kinematics_dynamics
```

---

## 3) (권장) orocos_kdl(C++) 먼저 빌드/설치

> PyKDL은 내부적으로 KDL C++ 라이브러리를 링크하므로, 먼저 설치해두면 안정적입니다.
> 

```bash
cd ~/Downloads/orocos_kinematics_dynamics

mkdir -p build_kdl && cd build_kdl
cmake ../orocos_kdl \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j
make install
```

설치 확인(선택):

```bash
ls $HOME/.local/lib | grep kdl || true
ls $HOME/.local/include | grep kdl || true
```

---

## 4) PyKDL(python_orocos_kdl)을 Isaac Python(3.11) 기준으로 빌드

### 4-1. Isaac Python에서 pybind11 includes 가져오기

```bash
export PYBIND11_INCLUDES=$($ISAAC_PY -m pybind11 --includes)
echo "$PYBIND11_INCLUDES"
```

### 4-2. CMake 빌드 (중요: 시스템 /usr/include/pybind11을 쓰지 않게 강제)

```bash
cd ~/Downloads/orocos_kinematics_dynamics/python_orocos_kdl
rm -rf build
mkdir build && cd build

export PYBIND11_CMAKEDIR=$($ISAAC_PY -m pybind11 --cmakedir)

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/.local" \
  -DPython3_EXECUTABLE="$ISAAC_PY" \
  -Dpybind11_DIR="$PYBIND11_CMAKEDIR" \
  -DCMAKE_CXX_FLAGS="$PYBIND11_INCLUDES"

make -j
```

### 왜 `DCMAKE_CXX_FLAGS="$PYBIND11_INCLUDES"`가 중요한가?

- 에러가 났던 상황은 컴파일 include가 `/usr/include/pybind11`로 잡혀서 **구버전 pybind11**이 들어간 케이스
- Python 3.11에서는 프레임 API/타입 노출이 달라서 구버전 pybind11이 자주 깨짐
- 그래서 Isaac Python(3.11)에 설치한 pybind11 include를 **우선순위 최상단**으로 강제합니다.
- 추가로, 이 저장소의 `python_orocos_kdl/CMakeLists.txt`는 `find_package(pybind11)`이 실패하면 내부 `pybind11/` 서브디렉터리를 기대합니다.
- 따라서 `-Dpybind11_DIR="$PYBIND11_CMAKEDIR"`를 같이 넘겨서 Isaac Python에 설치된 pybind11의 CMake 설정 경로를 명시하는 것이 안전합니다.

---

## 5) 빌드 결과(.so) 확인 & import 테스트

### 5-1. `.so` 생성 위치 찾기

```bash
cd ~/Downloads/orocos_kinematics_dynamics/python_orocos_kdl/build
find . -name "*.so" | head
```

### 5-2. Isaac Python으로 import 테스트

빌드 결과가 “현재 build 폴더 안”에 만들어지는 경우가 많아서 임시로 PYTHONPATH를 추가합니다.

```bash
# build 내부/혹은 패키지 폴더 구조에 따라 경로가 다를 수 있어 find로 확인한 위치를 사용
export PYTHONPATH=$PWD:$PYTHONPATH

$ISAAC_PY -c "import PyKDL; print('OK:', PyKDL.__file__)"
```

> import가 안 되면 `find . -name "PyKDL*.so"`로 정확한 폴더를 찾아서 그 폴더를 PYTHONPATH에 넣어주세요.
> 

---

## 6) (선택) 영구 설치 방식

### 방식 A: Isaac Python site-packages에 복사

가장 단순하지만, Isaac Sim 업데이트/재설치 시 날아갈 수 있습니다.

```bash
SITE_PKGS=$($ISAAC_PY -c "import sysconfig; print(sysconfig.get_paths()['platlib'])")
echo $SITE_PKGS

# 예: PyKDL*.so 위치가 ./build/PyKDL.so 라고 가정
cp <PyKDL.so_경로> $SITE_PKGS/
```

확인:

```bash
$ISAAC_PY -c "import PyKDL; print(PyKDL.__file__)"
```

### 방식 B: 별도 install prefix + PYTHONPATH 관리(권장)

내 홈 디렉터리 아래에 모듈을 두고 `PYTHONPATH`로 관리합니다.

```bash
mkdir -p $HOME/.local/isaac_py_modules
cp <PyKDL.so_경로> $HOME/.local/isaac_py_modules/

# 쉘 설정에 추가(~/.bashrc 등)
export PYTHONPATH=$HOME/.local/isaac_py_modules:$PYTHONPATH
```

---

## 7) 트러블슈팅 체크리스트

### 7-1. “/usr/include/pybind11”이 또 잡히는지 확인

```bash
cd ~/Downloads/orocos_kinematics_dynamics/python_orocos_kdl/build
make VERBOSE=1 -j |& tee build_verbose.log
grep -n "/usr/include/pybind11" build_verbose.log || true
```

- 잡힌다면: `DCMAKE_CXX_FLAGS="$PYBIND11_INCLUDES"`가 제대로 들어갔는지 확인

### 7-2. `pybind11.get_include()`가 없다고 나옴

- pybind11이 “제대로 pip 설치된 파이썬 패키지”가 아니었던 케이스
- 해결: `python -m pybind11 --includes` 방식으로 진행 (본 문서 방식)

### 7-3. 다음 단계에서 Boost 관련 링크 에러가 나는 경우

- PyKDL이 Boost.Python 기반으로 링크되는 구성이라면, Python 3.11용 Boost.Python이 필요할 수 있음
- 그때 나오는 링크 에러 메시지를 기준으로 Boost를 3.11로 맞추는 작업이 추가될 수 있음

---

# 최종 재현 커맨드 요약(복붙용)

```bash
export ISAAC_SIM=~/isaacsim-5.1.0
export ISAAC_PY=$ISAAC_SIM/python.sh

$ISAAC_PY -m pip install -U pip setuptools wheel
$ISAAC_PY -m pip install -U "pybind11>=2.10.4"

cd ~/Downloads
git clone https://github.com/orocos/orocos_kinematics_dynamics.git
cd orocos_kinematics_dynamics

mkdir -p build_kdl && cd build_kdl
cmake ../orocos_kdl -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j
make install
cd ..

export PYBIND11_INCLUDES=$($ISAAC_PY -m pybind11 --includes)

cd python_orocos_kdl
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$HOME/.local" -DPython3_EXECUTABLE="$ISAAC_PY" -DCMAKE_CXX_FLAGS="$PYBIND11_INCLUDES"
make -j

export PYTHONPATH=$PWD:$PYTHONPATH
$ISAAC_PY -c "import PyKDL; print('OK:', PyKDL.__file__)"
```

# 완성된  SO 파일 옮기기

build한 폴더를 들어가면 다음과 같음

![image.png](Isaac%20Sim%205%201%200/image.png)

해당 파일 PyKDL.cpython-… 파일을 직접 복사해서 다음에 위치시키면 됨

![image.png](Isaac%20Sim%205%201%200/image%201.png)