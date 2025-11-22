#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>


inline uint32_t _byteswap_uint32 (uint32_t n) {
    return 
    ((n & 0xFF000000) >> 24) |
    ((n & 0x00FF0000) >>  8) |
    ((n & 0x0000FF00) <<  8) |
    ((n & 0x000000FF) << 24);
}

struct NeuralData
{
    std::vector<std::vector<std::vector<double>>> images;
    // std::vector<std::vector<std::vector<uint8_t>>> images_uint8_t;
    std::vector<int> labels;

    NeuralData(std::string img, std::string nameLabels) {

        
        uint8_t byte{};
        uint32_t bytes4{};
        uint32_t dim{};
        int dims{};
        //read images
        std::ifstream imgfile(img, std::ios::binary);

        if(imgfile)std::cout<<"found images file.\n";
        imgfile.read(reinterpret_cast<char*>(&bytes4),4); // "magic number"
        
        imgfile.read(reinterpret_cast<char*>(&dim),4);    // first dimension (which means amount of images)
        
        dim = _byteswap_uint32(static_cast<uint64_t>(dim));

        std::cout<<dim<<"\n";
        
        
        images = std::vector<std::vector<std::vector<double>>>(
            dim, std::vector<std::vector<double>>(28,std::vector<double>(28, 0))
        );

        imgfile.read(reinterpret_cast<char*>(&bytes4),4); //other 2 dimensions (size of image) but they're always 28x28
        imgfile.read(reinterpret_cast<char*>(&bytes4),4);
        
        for(int i=0; i<dim; ++i) {
            for(int y=0; y<28; ++y) {
                for(int x=0; x<28; ++x) {
                    imgfile.read(reinterpret_cast<char*>(&byte),1);
                    //255 - 1
                    //0   - 0
                    images[i][y][x] = (double)byte/255.0;
                }
            }
        }
        
        imgfile.close();

        //read labels
        
        std::ifstream lblfile(nameLabels, std::ios::binary);
        
        if(lblfile)std::cout<<"found labels file.\n";
        
        lblfile.read(reinterpret_cast<char*>(&bytes4),4); // "magic number"
        
        lblfile.read(reinterpret_cast<char*>(&dim),4);    // first (and only) dimension
        
        labels = std::vector<int>(dim,0);
    
        
        for(int i=0; i<dim; ++i) {
            lblfile.read(reinterpret_cast<char*>(&byte),1);
            labels[i] = byte;
        }
        
        lblfile.close();
        
    }

    void display(int index) {
        std::cout<<"\n";
        std::cout<<labels[index];
        std::cout<<"\n";
        for(int y=0; y<28; ++y) {
            for(int x=0; x<28; ++x) {
                if(images[index][y][x]) {
                    if(images[index][y][x] > 0.75) {
                        std::cout<<"##";
                    } else if(images[index][y][x] > 0.5) {
                        std::cout<<"++";
                    } else if(images[index][y][x] > 0.25) {
                        std::cout<<"--";
                    } else {
                        std::cout<<"..";
                    }
                } else std::cout<<"  ";
            } std::cout<<"\n";
        }
        std::cout<<"\n";
        std::cout<<"\n";
    }
};




int main(int argc, char const *argv[])
{
    
    std::string labelsName = "t10k-labels.idx1-ubyte";
    std::string imagesName = "t10k-images.idx3-ubyte";

    NeuralData data(imagesName, labelsName);

    data.display(0);
    data.display(1);

    return 0;
}