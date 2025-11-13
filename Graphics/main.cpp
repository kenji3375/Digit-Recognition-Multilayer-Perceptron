#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>


#define PIXELS      28
#define PSIZE       280

#define HEIGHT      280
#define WIDTH       560


const sf::Font font("./minecraft_font.ttf");


class Pixels {
    
    public:
    std::vector<std::vector<bool>> mesh;

    Pixels() {
        mesh = std::vector<std::vector<bool>>(PIXELS,std::vector<bool>(PIXELS, false));
    }
    
    void clear() {
        mesh = std::vector<std::vector<bool>>(PIXELS,std::vector<bool>(PIXELS, false));
    }

    void setPixel(sf::Vector2i pos, sf::RenderWindow & window) {
        // if(pos.x>=0 && pos.x<PSIZE && pos.y >= 0 && pos.y<PSIZE) {
        //     std::cout<<pos.x*PIXELS / PSIZE<<std::endl;
        //     mesh[pos.y*PIXELS / PSIZE][pos.x*PIXELS / PSIZE] = true;
        // }

        if(pos.x>=0 && pos.x<window.getSize().x/2 && pos.y >= 0 && pos.y<window.getSize().y) {
            // std::cout<<pos.x*PIXELS / window.getSize().x/2<<std::endl;
            std::cout<<window.getSize().x/2<<std::endl;
            mesh[pos.y*PIXELS / (window.getSize().y)][pos.x*PIXELS / (window.getSize().x/2)] = true;
        }
    }
    void unsetPixel(sf::Vector2i pos, sf::RenderWindow & window) {
        if(pos.x>=0 && pos.x<window.getSize().x/2 && pos.y >= 0 && pos.y<window.getSize().y) {
            mesh[pos.y*PIXELS / (window.getSize().y)][pos.x*PIXELS / (window.getSize().x/2)] = false;
        }
    }

    void draw(sf::RenderWindow & window) {
        sf::RectangleShape rect;
        
        rect.setSize(sf::Vector2f(PSIZE,PSIZE));
        rect.setFillColor(sf::Color::Black);
        rect.setOutlineColor(sf::Color::White);
        rect.setOutlineThickness(2);
        
        window.draw(rect);
        
        rect.setOutlineThickness(0);
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




class Result {
    public:

    int result;
    float confidence;

    sf::Text resText;
    sf::Text confText;

    Result() :
        resText(font, "1"),
        confText(font, "0.00%")
    {
        
        result = -1;
        confidence = 0;
        

        resText.setCharacterSize(80);
        resText.setPosition(sf::Vector2f(390, 125));
        resText.setStyle(sf::Text::Bold);
        resText.setFillColor(sf::Color::White);
        
        confText.setCharacterSize(30);
        confText.setPosition(sf::Vector2f(370, 225));
        confText.setStyle(sf::Text::Bold);
        confText.setFillColor(sf::Color::White);

        unsetResult();
    }

    void draw(sf::RenderWindow & window) {
        window.draw(resText);
        window.draw(confText);
    }

    void getResult(Pixels pixels) {
        
        //getting results will be made here
        if(pixels.mesh[0][0]) {
            result = 1;
            confidence = 30.57;
        }
        //turning the results into text
        std::ostringstream oss;

        oss << std::fixed << std::setprecision(1)<<result;
        
        resText.setString(oss.str());
        
        oss.str("");
        
        oss << std::fixed << std::setprecision(2)<<confidence<<"%";
        confText.setString(oss.str());
        
        std::cout<<"res set";
    }
    
    void unsetResult() {
        resText.setString("");
        confText.setString("00.00%");
    }

};


int main()
{
    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "SFML works!");
    // sf::CircleShape shape(100.f);
    // shape.setFillColor(sf::Color::Green);

    Pixels pixels;
    Result result;

    bool leftDown  = false;
    bool rightDown = false;
    bool spaceDown = false;


    
    // Create a text
    sf::Text text(font, "LMB - draw\nRMB - remove pixel\nSpace - clear all\nEscape - exit\nResult:");
    text.setCharacterSize(15);
    text.setStyle(sf::Text::Bold);
    text.setFillColor(sf::Color::White);
    text.setOutlineColor(sf::Color::White);

    text.setPosition(sf::Vector2f(300,20));
    
    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>() || sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) {
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
                result.getResult(pixels);
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
            

            //spacebar key
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space) && !spaceDown) {
                std::cout<<"spacebar pressed\n";
                spaceDown = true;
            } 
            if(spaceDown && !sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space)) {
                spaceDown = false;
                std::cout<<"spacebar unpressed\n";
            }

        }
        
        if(leftDown) {
            pixels.setPixel(sf::Mouse::getPosition(window), window);
        } else if(rightDown) {
            pixels.unsetPixel(sf::Mouse::getPosition(window), window);
        }

        if(spaceDown) {
            pixels.clear();
            result.unsetResult();
        }
        
        window.clear();
        
        pixels.draw(window);
        result.draw(window);
        window.draw(text);
        
        window.display();
    }
}