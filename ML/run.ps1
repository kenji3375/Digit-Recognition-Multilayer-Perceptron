#g++ main.cpp -std=c++17 -IC:\msys64\ucrt64\include -LC:\msys64\ucrt64\lib -lsfml-graphics -lsfml-window -lsfml-system -o main.exe
# g++ main.cpp -std=c++17 -DSFML_STATIC -IC:/msys64/ucrt64/include -LC:/msys64/ucrt64/lib -lsfml-graphics-s -lsfml-window-s -lsfml-system-s -lfreetype -lopengl32 -lgdi32 -lwinmm -lpthread -static -static-libgcc -static-libstdc++ -o main.exe
g++ main.cpp -std=c++17 -DSFML_STATIC -IC:/msys64/ucrt64/include -LC:/msys64/ucrt64/lib -lsfml-graphics-s -lsfml-window-s -lsfml-system-s -lfreetype -lharfbuzz -lgraphite2 -lpng16 -lbrotlidec -lbrotlicommon -lz -lbz2 -lopengl32 -lgdi32 -lwinmm -ldwrite -lrpcrt4 -lpthread -static -static-libgcc -static-libstdc++ -o main.exe


./main.exe