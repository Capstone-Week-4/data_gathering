- vscode를 사용하시는 경우 `ctrl + shift + D` 하셔서 디버깅으로 실행해주셔야 합니다.
- requirements.txt 패키지를 내려받아 사용해 주세요.
  <br>

### 실행
```
INSERT THE NUMBER OF SIGNS : 20
```
- 학습시키고자 하는 단어의 개수를 입력합니다.
- 위 input에는 한개의 테마에 속하는 단어의 개수를 입력해주세요. (예 : 운동 테마 20단어인 경우 20 입력)

<br>

```
[1]INSERT THE VOCABULARY EACH : first vocab
[2]INSERT THE VOCABULARY EACH : second vocab
[3]INSERT THE VOCABULARY EACH : third vocab
```
- 지시에 따라 단어들을 하나씩 입력해주세요.
- 카카오톡으로 보내주신 단어 목록의 순서를 잘 지켜주셔야 합니당~

<br>
```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
```
- 위와 같은 출력이 터미널에 뜨면, 아래 작업표시줄에 새로운 창이 생성됩니다.
- 창을 클릭하여 수화 동작을 하며 대기해주세요 (화면에 "Take {action} please" 라고 뜬 <b>정지화면</b> 상태에서 대기해주세요.)
- 화면에 상체와 두 손이 모두 인식되도록 카메라를 조정해주세요. 손이 두쪽 다 인식되지 않으면 데이터가 수집되지 않습니다...!! ㅠㅠ (그렇다고 중간에 몇 프레임 빠졌다고 다시 시작하실 필요는 없습니다...)
- 한개의 수화에 대한 수집이 완료되면 터미널창에 몇 행의 데이터가 수집되었는지 출력됩니다. 모든 수화 단어를 수집한 후 터미널창을 확인하여 데이터가 고르게 수집되었는지 확인해주세요.
- <b>한번 코드를 재시작하면 `/dataset` 파일 내의 모든 데이터셋이 일괄 삭제됩니다...!! 대참사가 일어나지 않도록 주의해주세요...</b>
