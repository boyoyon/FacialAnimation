<html lang="ja">
    <head>
        <meta charset="utf-8" />
    </head>
    <body>
        <h1><center>Facial Amimation</center></h1>
        <h2>なにものか？</h2>
        <p>
            顔の静止画像に表情をつけます。<br>
            Real_Time_Image_Animationの学習済モデルパラメータのリンクが切れていたり、依存するライブラリが多かったので、最低限のライブラリで動作するようにしてみました。<br>
            <br>
            (入力)
            <img src="images/input.png"><br>
            (出力)
            <img src="images/output.gif"><br>
        </p>
        <h2>環境構築方法</h2>
        <p>
            <h3>[1] ベース環境をダウンロード～解凍～配置する</h3>
            　<a href="https://github.com/anandpawara/Real_Time_Image_Animation">Real time Image Animation</a><br>
            　Code → Download ZIP をクリックする。<br>
            <br>
            　Real_Time_Image_Animation.zip を解凍し、Real_Time_Image_Animation フォルダ内のファイル、フォルダを src フォルダの下に配置する。<br>
            <br>
            <h3>[2] 学習済モデルパラメータをダウンロード～配置する</h3>
            　<a href="https://disk.yandex.ru/d/lEw8uRm140L_eQ">https://disk.yandex.ru/d/lEw8uRm140L_eQ</a><br>
            　から vox-cpk.pth.tar をダウンロードし、srcフォルダの下に配置する。<br>
            <br>
            <h3>[3] 表情を駆動する画像群を配置する</h3>
            　表情を駆動する画像群(.png)を driving_images フォルダの下に配置する。<br>
            　animate.gif のヒゲのおじさんを駆動画像に使用する場合は<br>
            　python create_driving_images.py <br>
            　でフォルダの作成～画像の切り出しが実行される。<br>
            <br>
            <h3>[4] ライブラリをインストールする</h3>
            　・PyTorchをインストールする<br>
            　　手持ちのGPUの都合でv1.13.0でしか試しておりません。<br>
            　　<a href="https://pytorch.org/get-started/previous-versions/">https://pytorch.org/get-started/previous-versions/</a><br>
            　　v1.13.0 のpip install の手順を参照。<br>
            　・pip install scipy<br>
            　・pip install pyyaml<br>
            　・pip install opencv-python<br>
            　・pip install numpy==1.26.1<br>
        </p>
        <h2>使い方</h2>
        <p>
            <h3>Facial Animation 画像群生成</h3>
            　python facial_animation.py (顔画像)<br>
            　result フォルダに画像群が出力されます。<br>
            　遅くても構わなければ、GPU無しでも動作します。<br>
            　qキーを何回か押すと画像生成を中断できます。<br>
            <br>
            <h3>gifアニメーション化</h3>
            　imageio がインストールされていない場合は、pip install imageio<br>
            　python images_to_gif.py result\*.png (FPS) (ループ回数)<br>
            　output.gif が出力されます。<br>
            　ループ回数に0を指定すると無限にループします。<br>
            <br>
            <h3>mp4化</h3>
            　python images_to_mp4.py result\*.png<br>
            　output.mp4 が出力されます。<br> 
        </p>
    </body>
</html>
