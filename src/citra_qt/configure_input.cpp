#include "configure_input.h"
#include "ui_configure_input.h"

ConfigureInput::ConfigureInput(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ConfigureInput)
{
    ui->setupUi(this);
}

ConfigureInput::~ConfigureInput()
{
    delete ui;
}
