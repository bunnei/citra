#ifndef CONFIGURE_INPUT_H
#define CONFIGURE_INPUT_H

#include <QWidget>

namespace Ui {
class ConfigureInput;
}

class ConfigureInput : public QWidget
{
    Q_OBJECT

public:
    explicit ConfigureInput(QWidget *parent = 0);
    ~ConfigureInput();

private:
    Ui::ConfigureInput *ui;
};

#endif // CONFIGURE_INPUT_H
