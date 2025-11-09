#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>

#define HEIGHT      280
#define WIDTH       280



class Pixels {
    std::vector<std::vector<bool>> mesh;
    
    public:

    Pixels() {
        mesh = std::vector<std::vector<bool>>(28,std::vector<bool>(28, false));
    }

    void setPixel(sf::Vector2i pos) {
        if(pos.x >= 0 && pos.x<=WIDTH && pos.y >= 0 && pos.y<=HEIGHT) {
            mesh[pos.y/10][pos.x/10] = true;
        }
    }
    void unsetPixel(sf::Vector2i pos) {
        if(pos.x >= 0 && pos.x<=WIDTH && pos.y >= 0 && pos.y<=HEIGHT) {
            mesh[pos.y/10][pos.x/10] = false;
        }
    }

    void draw(sf::RenderWindow & window) {
        sf::RectangleShape rect;
        rect.setSize(sf::Vector2f(10,10));
        rect.setFillColor(sf::Color::White);
        rect.setOutlineColor(sf::Color::Black);

        for(int y=0; y<28; ++y) {
            for(int x=0; x<28; ++x) {
                if(mesh[y][x]) {
                    rect.setPosition(sf::Vector2f(x*10, y*10));
                    window.draw(rect);
                }
            }
        }


    }

    ~Pixels() = default;
};




int main()
{
    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "SFML works!");
    // sf::CircleShape shape(100.f);
    // shape.setFillColor(sf::Color::Green);

    Pixels pixels;

    bool leftDown=false;
    bool rightDown=false;
    
    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            
            
            
            //left mouse buttons
            if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left) && !leftDown) {
                std::cout<<"mouse pressed\n";
                leftDown = true;
                
                // std::cout<<sf::Mouse::getPosition(window).x<<"    "<< sf::Mouse::getPosition(window).y<<"\n";
                
            } 
            if(leftDown && !sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
                leftDown = false;
                std::cout<<"mouse unpressed\n";
            }
            
            //right mouse buttons
            if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Right) && !rightDown) {
                std::cout<<"r mouse pressed\n";
                rightDown = true;
            } 
            if(rightDown && !sf::Mouse::isButtonPressed(sf::Mouse::Button::Right)) {
                rightDown = false;
                std::cout<<"r mouse unpressed\n";
            }
        }
        
        if(leftDown) {
            pixels.setPixel(sf::Mouse::getPosition(window));
        } else if(rightDown) {
            pixels.unsetPixel(sf::Mouse::getPosition(window));
        }
        
        window.clear();
        
        pixels.draw(window);
        
        window.display();
    }
}