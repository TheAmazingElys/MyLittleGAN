import torch
import torch.nn as nn
import torch.optim as optim

REAL_LABEL = 1.0
FAKE_LABEL = 0.0


def train(
    gen_net,
    disc_net,
    dataloader,
    device,
    nb_epochs=20,
    lr=0.0002,
    loss_function=nn.BCELoss(),
):

    assert dataloader.drop_last  # We assume the last batch to be dropped if incomplete
    disc_optimizer = optim.Adam(disc_net.parameters(), lr=lr, betas=(0.5, 0.999))
    gen_optimizer = optim.Adam(gen_net.parameters(), lr=lr, betas=(0.5, 0.999))

    iters = 0

    print("Starting Training Loop...")
    for i_epoch in range(nb_epochs):
        for i, i_batch in enumerate(dataloader, 0):

            """
            Training of the discriminator with actual data
            """
            disc_net.zero_grad()
            label = torch.full(
                (batch_size,), REAL_LABEL, dtype=torch.float, device=device
            )

            """When generalizing the code to others dataset we should do the unsqueeze in the collate_fn"""
            output = disc_net(i_batch.to(self.device).float().unsqueeze(1)).view(-1)
            loss_real = loss_function(output, label)
            loss_real.backward()

            D_x = output.mean().item()

            """
            Training of the discriminator with all fake data
            """
            fake = gen_net(gen_net.get_noise(device, batch_size))
            label.fill_(FAKE_LABEL)

            output = disc_net(fake.detach()).view(-1)
            loss_fake = loss_function(output, label)
            loss_fake.backward()

            loss_disc = loss_real.item() + loss_fake.item()

            disc_optimizer.step()

            """
            Training the generator to make better fakes
            """

            gen_net.zero_grad()
            label.fill_(
                REAL_LABEL
            )  # From the generator side, it must generate real imgs

            output = disc_net(fake).view(-1)

            loss_gen = loss_function(output, label)
            loss_gen.backward()

            gen_optimizer.step()

            """
            Print progress of training
            """
            if i % 50 == 0:
                print(
                    f"[{i_epoch}/{nb_epochs}][{i}/{len(dataloader)}] Loss_D: {round(loss_disc, 3)} Loss_G {round(loss_gen.item(), 3)} D(x): {round(D_x, 3)}"
                )

            if (iters % 500 == 0) or (
                (i_epoch - 1 == nb_epochs) and (i == len(dataloader) - 1)
            ):
                """
                Save the fakes each 500 iterations and at the end of the training
                """
                with torch.no_grad():
                    fake = gen_net(gen_net.get_fixed_noise(device)).detach().cpu()
                    utils.plot_img(fake, file_name=f"fake_{iters}.jpg")

            iters += 1
